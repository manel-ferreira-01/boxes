import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
import json

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
        self.output_folder = os.getenv("OUTPUT_FOLDER", "/data/output")
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        self.lock = threading.Lock()
        self.last_envelope = None

        # --- video-specific state ---
        self.video_frames = []
        self.frame_index = 0
        self.active_video = None
        self.video_mode = False

        # --- watchdog setup ---
        event_handler = FileHandler(lambda path: self._schedule_process(path))
        self.observer = Observer()
        self.observer.schedule(event_handler, self.input_folder, recursive=False)
        self.observer.start()
        logging.info(f"Watching folder: {self.input_folder}")

    # ------------------------------------------------------------------
    # --- Dispatch based on file type
    # ------------------------------------------------------------------
    def _schedule_process(self, path):
        if path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            logging.info("Scheduling video processing")
            threading.Thread(target=self._process_video, args=(path,), daemon=True).start()
        elif path.lower().endswith((".jpg", ".png")):
            logging.info("Scheduling image batch processing")
            # if no videos in the folder, just process images
            threading.Thread(target=self._process_images_batch, daemon=True).start()

    # ------------------------------------------------------------------
    # --- Video handling
    # ------------------------------------------------------------------
    def _process_video(self, video_path):
        """Extract frames and prepare first frame for processing."""
        logging.info(f"Extracting frames from video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(buf.tobytes())
        cap.release()

        with self.lock:
            self.video_frames = frames
            self.frame_index = 0
            self.active_video = os.path.basename(video_path)
            self.video_mode = True
            self._prepare_next_frame()

        logging.info(f"Loaded {len(frames)} frames from {video_path}")

    def _prepare_next_frame(self):
        """Prepare the next frame as an envelope."""
        if self.frame_index >= len(self.video_frames):
            logging.info("All video frames processed.")
            self.last_envelope = None
            self.video_mode = False
            return

        frame_bytes = self.video_frames[self.frame_index]
        
        annotations = {
            "parameters": {"ssim_thresh": 0.70, "blur_kernel": 5},
            "frame_index": self.frame_index,
            "video_name": self.active_video,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        logging.info(f"Prepared frame {self.frame_index} of video {self.active_video}")
        self.last_envelope = folder_wd_pb2.Envelope(
            config_json=json.dumps({"opencv": annotations}),
            data={"images": wrap_value([frame_bytes])},
        )

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
            "parameters": {"ssim_thresh": 0.90, "blur_kernel": 5},
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

            # --- Validate we got a usable response ---
            if changed is None:
                #logging.warning(f"No 'changed' flag in response: {out_json}")
                return folder_wd_pb2.Empty()

            # --- Log SSIM and decision ---
            if changed:
                logging.info(f"Frame {self.frame_index}: CHANGE detected (SSIM={ssim_val:.3f}). Saving frame.")
            else:
                logging.info(f"Frame {self.frame_index}: No change (SSIM={ssim_val:.3f}). Skipping save.")

            # --- Save frame only if change detected ---
            if changed:
                images = unwrap_value(response.data.get("images")) if response.data else []
                if images:
                    nparr = np.frombuffer(images[0], np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    out_path = os.path.join(self.output_folder, f"processed_{self.frame_index:05d}.jpg")
                    cv2.imwrite(out_path, img)
                    logging.info(f"Saved frame at: {out_path}")

            # --- Advance to next frame regardless of change decision ---
            if self.video_mode:
                self.frame_index += 1
                self._prepare_next_frame()

        except Exception as e:
            logging.error(f"Display error: {e}")

        return folder_wd_pb2.Empty()
    
    def display_opencv(self, response, context):
        try:
            
            in_json = json.loads(response.config_json)
            if in_json.get("changed") == True:
                logging.info("got an image")

        except Exception as e:
            pass
        
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
