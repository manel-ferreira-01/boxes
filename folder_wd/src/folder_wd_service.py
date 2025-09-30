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
    def __init__(self, process_fn,  settle_time=1.5):
        self.process_fn = process_fn
        self.settle_time = settle_time

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith((".jpg",".png")):
            logging.info(f"New image detected: {event.src_path}")
            if self.wait_until_stable(event.src_path):
                self.process_fn(event.src_path)
            else:
                logging.warning(f"File {event.src_path} not stable, skipped.")
    
    def wait_until_stable(self,path, check_interval=0.5, max_checks=10):
        """Wait until file size stops changing, return True if stable."""
        last_size = -1
        for _ in range(max_checks):
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                return False  # file vanished
            if size == last_size:
                return True   # stable
            last_size = size
            time.sleep(check_interval)
        return False



class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self.input_folder = os.getenv('INPUT_FOLDER', '/data')
        self.output_folder = os.getenv('OUTPUT_FOLDER', '/data/output')
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        logging.info(f"Input folder: {self.input_folder}")

        self.input_count = 0
        self.last_envelope = None
        self.lock = threading.Lock()
        self.debounce_timer = None
        self.debounce_seconds = 2.0

        # process startup files if present
        if any(f.lower().endswith((".jpg", ".png")) for f in os.listdir(self.input_folder)):
            self._schedule_batch()

        # watchdog observer
        event_handler = FileHandler(lambda path: self._schedule_batch())
        self.observer = Observer()
        self.observer.schedule(event_handler, self.input_folder, recursive=False)
        self.observer.start()

    def _schedule_batch(self):
        """(Re)start debounce timer to process folder as a batch."""
        if self.debounce_timer:
            self.debounce_timer.cancel()
        self.debounce_timer = threading.Timer(self.debounce_seconds, self._process_folder)
        self.debounce_timer.start()

    def _process_folder(self):
        """Actually read all images in folder and build Envelope."""
        image_files = [f for f in os.listdir(self.input_folder)
                       if f.lower().endswith((".jpg", ".png"))]
        if not image_files:
            return

        images = []
        for fname in sorted(image_files):  # deterministic order
            path = os.path.join(self.input_folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            _, buf = cv2.imencode(".jpg", img)
            images.append(buf.tobytes())
            #os.remove(path) # delete the imageq after reading

        self.input_count += 1
        annotations = {
            "command": "detectsequence",
            "input_count": self.input_count,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        envelope = folder_wd_pb2.Envelope(
            config_json=json.dumps({"aispgradio": annotations}),
            data={"images": wrap_value(images)},
        )

        with self.lock:
            self.last_envelope = envelope

        logging.info(f"Prepared new batch of {len(images)} images")

    def acquire(self, request, context):
        """Return prepared batch if available, else empty."""
        
        with self.lock:
            env = self.last_envelope

        time.sleep(0.1)  # slight delay to allow batch to be set

        if env:
            # consume the envelope once (optional: clear it)
            with self.lock:
                self.last_envelope = None
            return env
        else:
            # reply immediately with an "empty" message
            out_json = json.dumps({"aispgradio": {"empty": "empty"}})
            _, tmp = cv2.imencode(".jpg", np.zeros((2, 2, 3), dtype="uint8"))
            return folder_wd_pb2.Envelope(
                config_json=out_json,
                data={"images": wrap_value([tmp.tobytes()])}
            )

        
    def display(self, response, context):
        """Receive processed results, save images to output folder."""
        time.sleep(0.1)  # slight delay to allow batch to be set
        try:
            images = unwrap_value(response.data["images"])
            #logging.info(f"Received {len(images)} processed images")
            if not images:
                return folder_wd_pb2.Empty()

            saved_files = []
            for idx, img_bytes in enumerate(images, start=1):
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    logging.warning(f"Could not decode image {idx}")
                    continue

                out_path = os.path.join(self.output_folder, f"processed_{idx:03d}.jpg")
                cv2.imwrite(out_path, img)
                saved_files.append(out_path)

            # save metadata/detections
            meta_path = os.path.join(self.output_folder, "detections.json")
            with open(meta_path, "w") as f:
                f.write(response.config_json)

        except Exception as e:
            logging.error(f"Error processing received data: {e}")

        return folder_wd_pb2.Empty()

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
