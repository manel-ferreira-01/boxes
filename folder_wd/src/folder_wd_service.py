import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
import json
import numpy as np
import pickle

# Import proto files and auxiliary functions
import sys
sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2
import pipeline_pb2_grpc as folder_wd_pb2_grpc
from aux import wrap_value, unwrap_value

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = 'PORT'

#specfic imports for the service
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import cv2
import datetime
import numpy as np
import zstandard as zstd
import io
import traceback
from queue import Queue

def numpy_bytes_to_data(b: bytes) -> np.ndarray:
    buf = io.BytesIO(b)
    arr = np.load(buf, allow_pickle=False)
    return arr



class FileHandler(FileSystemEventHandler):
    """Watches the input folder for new image or video files."""
    def __init__(self, process_fn, settle_time=1.5):
        self.process_fn = process_fn
        self.settle_time = settle_time

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith((".jpg", ".png", ".mp4", ".avi", ".mov", ".mkv")):
            logging.info(f"New file detected: {event.src_path}")
            if self._wait_until_stable(event.src_path):
                self.process_fn(event.src_path)
            else:
                logging.warning(f"File {event.src_path} not stable, skipped.")

    def _wait_until_stable(self, path, check_interval=0.5, max_checks=10):
        last_size = -1
        for _ in range(max_checks):
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                return False
            if size == last_size:
                return True
            last_size = size
            time.sleep(check_interval)
        return False


class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self.input_folder = os.getenv("INPUT_FOLDER", "/data")
        self.base_output_folder = os.getenv("OUTPUT_FOLDER", "/data/output")
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.base_output_folder, exist_ok=True)

        self.lock = threading.Lock()
        self.last_envelope = None

        # --- video-specific state ---
        self.video_frames = []
        self.frame_index = 0
        self.active_video = None
        self.video_mode = False
        self.processed_frames = set()
        self.vggt_envelope = None

        # --- watchdog setup ---
        event_handler = FileHandler(lambda path: self._schedule_process(path))
        self.observer = Observer()
        self.observer.schedule(event_handler, self.input_folder, recursive=False)
        self.observer.start()
        logging.info(f"Watching folder: {self.input_folder}")

        self.videos = {}               # active videos {name: {"frames":[], "index":int}}
        self.video_queue = Queue()     # waiting queue
        threading.Thread(target=self._video_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # --- Dispatch based on file type
    # ------------------------------------------------------------------
    def _schedule_process(self, path):
        if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            logging.info(f"Queueing new video: {path}")
            self.video_queue.put(path)
        elif path.lower().endswith((".jpg", ".png")):
            threading.Thread(target=self._process_images_batch, daemon=True).start()

    def _video_worker(self):
        """Background thread that processes videos in sequence."""
        while True:
            path = self.video_queue.get()
            try:
                self._process_video(path)
                # --- Wait until video finishes before moving to next ---
                while self.active_video is not None:
                    time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error processing {path}: {e}")
            finally:
                self.video_queue.task_done()


    def _process_video(self, video_path, max_size=(1920, 1080)):
        """Extract frames, enqueue per-frame envelopes with stream flag."""
        logging.info(f"Extracting frames from video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        max_w, max_h = max_size
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if w > max_w or h > max_h:
                scale = min(max_w / w, max_h / h)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(buf.tobytes())
        cap.release()

        video_name = os.path.basename(video_path)
        self.output_folder = os.path.join(self.base_output_folder, os.path.splitext(video_name)[0])
        os.makedirs(self.output_folder, exist_ok=True)

        self.videos[video_name] = {"frames": frames, "index": 0, "output": self.output_folder}
        self.active_video = video_name

        # trigger sending first frame
        self._prepare_next_frame()
        logging.info(f"Loaded {len(frames)} frames for {video_name}")

    def _prepare_next_frame(self):
        """Prepare the next frame envelope for a specific video."""
        if not self.active_video or self.active_video not in self.videos:
            logging.warning("No active video to prepare next frame for.")
            return

        vid = self.videos[self.active_video]
        frames, idx = vid["frames"], vid["index"]
        total = len(frames)
        self.frame_index = idx
        if idx >= total:
            logging.info(f"Video {self.active_video} finished.")
            self.videos.pop(self.active_video, None)
            self.active_video = None
            return
        
        self.frame_bytes = frames[idx]
        countdown = total - idx - 1  # stream flag
        annotations = {
            "parameters": {"ssim_thresh": 0.70, "blur_kernel": 5, "device":"cuda:1",
                           "motion_thresh": 45},
            "frame_index": idx,
            "stream": countdown
        }

        with self.lock:
            self.last_envelope = folder_wd_pb2.Envelope(
                config_json=json.dumps({"opencv": annotations}),
                data={"images": wrap_value([self.frame_bytes])}
            )

        logging.info(f"Prepared frame {idx+1}/{total} for video {self.active_video} with stream={countdown}")

    # ------------------------------------------------------------------
    # --- Image batch handling
    # ------------------------------------------------------------------
    def _process_images_batch(self):
        """Process all standalone images in the folder as a batch."""
        image_files = [
            f for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".png"))
        ]
        if not image_files:
            return

        images = []
        for fname in sorted(image_files):
            path = os.path.join(self.input_folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            _, buf = cv2.imencode(".jpg", img)
            images.append(buf.tobytes())

        annotations = {
            "parameters": {"ssim_thresh": 0.90, "blur_kernel": 5, "motion_thresh": 45},
            "timestamp": datetime.datetime.now().isoformat(),
            "input_count": len(images),
        }

        with self.lock:
            self.last_envelope = folder_wd_pb2.Envelope(
                config_json=json.dumps({"opencv": annotations}),
                data={"images": wrap_value(images)},
            )
            self.video_mode = False

        logging.info(f"Prepared image batch of {len(images)} images")

    # ------------------------------------------------------------------
    # --- gRPC methods
    # ------------------------------------------------------------------
    def acquire(self, request, context):
        """Return next data packet to the next service."""
        time.sleep(0.05)
        with self.lock:
            env = self.last_envelope
        if env:
            # consume the envelope once (optional: clear it)
            with self.lock:
                self.last_envelope = None
            return env
        else:
            return folder_wd_pb2.Envelope()
    
    def acquire_vggt(self, request, context):
        #logging.info("acquire_vggt called")
        with self.lock:
            env = self.vggt_envelope
        if env:
            logging.info("Returning vggt_envelope")
            # consume the envelope once (optional: clear it)
            with self.lock:
                self.vggt_envelope = None
                logging.info("vggt_envelope consumed")
            return env
        else:
            #logging.info("No vggt_envelope to return")
            return folder_wd_pb2.Envelope()

    def display(self, response, context):
        """Receive OpenCV similarity_check results, decide whether to save and advance."""
        try:
            # --- Parse the JSON output safely ---
            try:
                out_json = json.loads(response.config_json) if response.config_json else {}
            except Exception:
                logging.error("Invalid JSON in display() response.")
                return folder_wd_pb2.Empty()

            changed = out_json.get("changed", None)
            ssim_val = out_json.get("ssim", None)
            metric = out_json.get("metric", None)

            # --- Validate we got a usable response ---
            if changed is None:
                #logging.warning(f"No 'changed' flag in response: {out_json}")
                return folder_wd_pb2.Empty()

            # --- Save frame only if change detected ---
            if changed:
                images = unwrap_value(response.data.get("images")) if response.data else []
                if images:
                    nparr = np.frombuffer(images[0], np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    out_path = os.path.join(self.output_folder, f"processed_{self.frame_index:05d}.jpg")
                    cv2.imwrite(out_path, img)
                    logging.info(f"metric: {metric}")
                    logging.info(f"Saved frame at: {out_path}")

            # --- Advance to next frame regardless of change decision ---
            if self.video_mode:
                self.frame_index += 1
                self._prepare_next_frame()

        except Exception as e:
            tb = traceback.format_exc()
            logging.error("[folder_wd_display] Unhandled exception:\n%s", tb)

        return folder_wd_pb2.Empty()

    def display_langsam(self, response, context):
        if not response.config_json:
            return folder_wd_pb2.Empty()
        else:
            try:
                out_json = json.loads(response.config_json)
                if out_json.get("status"):
                    if response.data.get("results", []):
                        logging.info("got results from langsam")
                        pkl = pickle.loads(zstd.decompress(unwrap_value(response.data["results"])))
                        # save pickle into output folder
                        out_path = os.path.join(self.output_folder, f"lang_sam/{self.frame_index:05d}.pkl")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True) # just to check
                        with open(out_path, "wb") as f:
                            pickle.dump(pkl, f)

                    if self.active_video and self.active_video in self.videos:
                        self.videos[self.active_video]["index"] += 1
                        self._prepare_next_frame()


                return folder_wd_pb2.Empty()
            except Exception as e:
                tb = traceback.format_exc()
                logging.error("[folder_wd_display_langsam] Unhandled exception:\n%s", tb)
                return folder_wd_pb2.Empty()
    

    def display_yolo(self, response, context):
        if not response.config_json:
            return folder_wd_pb2.Empty()
        else:
            try:
                out_json = json.loads(response.config_json)
                if out_json.get("YOLO"):
                    if response.data.get("images", []):
                        logging.info("got results from yolo")
                        images = unwrap_value(response.data["images"])
                        for idx, img_bytes in enumerate(images): # it should only be one img
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            out_path = os.path.join(self.output_folder, f"yolo/processed_{self.frame_index:05d}.jpg")
                            os.makedirs(os.path.dirname(out_path), exist_ok=True) # just to check
                            cv2.imwrite(out_path, img)

                        #also save self.frame_bytes
                        out_path_orig =  os.path.join(self.output_folder, f"original/orig_{self.frame_index:05d}.jpg")
                        nparr = np.frombuffer(self.frame_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        os.makedirs(os.path.dirname(out_path_orig), exist_ok=True) # just to check
                        cv2.imwrite(out_path_orig, img)
                        # save the frame idx used
                        self.processed_frames.add(self.frame_index)

                        # save the json from the detections
                        out_path_json = os.path.join(self.output_folder, f"yolo_json/detections_{self.frame_index:05d}.json")
                        os.makedirs(os.path.dirname(out_path_json), exist_ok=True) # just to check
                        with open(out_path_json, "w") as f:
                            json.dump(out_json["YOLO"], f)
                        
                    # ONLY THE LANGSAM PREPARES THE FRAMES, SINCE IT WOULD PROBABLY TAKES MORE TIME THAN YOLO

                return folder_wd_pb2.Empty()
            except Exception as e:
                tb = traceback.format_exc()
                logging.error("[folder_wd_display_yolo] Unhandled exception:\n%s", tb)
                return folder_wd_pb2.Empty()
    
    def display_vggt(self, response, context):
        if not response.config_json:
            return folder_wd_pb2.Empty()
        else:
            try:
                out_json = json.loads(response.config_json)
                if out_json.get("VGGT"):
                    if response.data.get("world_points", []):
                        logging.info("got results from VGGT")

                        out_dict = {}
                        out_dict["wrld_points"] = numpy_bytes_to_data(unwrap_value(response.data["world_points"]))
                        out_dict["world_points_conf"] = numpy_bytes_to_data(unwrap_value(response.data["world_points_conf"]))
                        out_dict["depth"] = numpy_bytes_to_data(unwrap_value(response.data["depth"]))
                        out_dict["depth_conf"] = numpy_bytes_to_data(unwrap_value(response.data["depth_conf"]))
                        out_dict["images"] = numpy_bytes_to_data(unwrap_value(response.data["images"]))

                        pickle_data = pickle.dumps(out_dict)
                        out_path = os.path.join(self.output_folder, f"vggt/{self.frame_index:05d}.pkl")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True) # just to check
                        logging.info(f"Saving vggt data at: {out_path}")

                        with open(out_path, "wb") as f:
                            f.write(pickle_data)      

                return folder_wd_pb2.Empty()
            except Exception as e:
                tb = traceback.format_exc()
                logging.error("[folder_wd_display_vggt] Unhandled exception:\n%s", tb)
                return folder_wd_pb2.Empty()
            
    


    # ------------------------------------------------------------------
    def stop(self):
        self.observer.stop()
        self.observer.join()


# ----------------------------------------
# Server setup and running
# ----------------------------------------

def get_port():
    """
    Parses the port where the server should listen
    Exists the program if the environment variable
    is not an int or the value is not positive

    Returns:
        The port where the server should listen or
        None if an error occurred

    """
    try:
        server_port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if server_port <= 0:
            logging.error('Port should be greater than 0')
            return None
        return server_port
    except ValueError:
        logging.exception('Unable to parse port')
        return None

def run_server(server):
    """Run the given server on the port defined
    by the environment variables or the default port
    if it is not defined

    Args:
        server: server to run

    """
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'''Server started at {target}''')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        

if __name__ == '__main__':
    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    #Create Server and add service
    server = grpc.server(futures.ThreadPoolExecutor(),
                         options= [('grpc.max_send_message_length', -1), 
                                   ('grpc.max_receive_message_length', -1)])
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    # Add reflection
    service_names = (
        folder_wd_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
