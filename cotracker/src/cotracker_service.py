import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import inspect
import os
import time
    
import io
from scipy.io import loadmat, savemat
import numpy as np

import torch

from PIL import Image
import torch.nn.functional as F
import tempfile
import decord


from importlib.machinery import SourceFileLoader
cotracker_pb2 = SourceFileLoader(
    "cotracker_pb2",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../protos/cotracker_pb2.py")
).load_module()
cotracker_pb2_grpc = SourceFileLoader(
    "cotracker_pb2_grpc",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../protos/cotracker_pb2_grpc.py")
).load_module()


_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

import threading
IDLE_TIMEOUT = 60  # seconds (1 min)

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.cpu(), buf)
    return buf.getvalue()

class CoTrackerService(cotracker_pb2_grpc.CoTrackerServiceServicer):

    def __init__(self):
        # Always load to CPU first
        self._model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cpu")
        logging.info("Model loaded on CPU")
        self._device = "cpu"

        self._last_request_time = time.time()
        self._lock = threading.Lock()

        # Background thread to monitor idle time
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        while True:
            time.sleep(10)  # check every 10s
            with self._lock:
                idle_time = time.time() - self._last_request_time
                if idle_time > IDLE_TIMEOUT and self._device == "cuda":
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"

    def Forward(self, request,context):
        with self._lock:
            self._last_request_time = time.time()

            # If idle watchdog moved it back to CPU, restore to GPU
            if self._device == "cpu" and torch.cuda.is_available():
                logging.info("Request received: moving model to GPU")
                self._model.to("cuda")
                self._device = "cuda"

        # Run inference
        tracks_tensor = run_codigo(request, self._model, self._device)

        response = cotracker_pb2.CoTrackerResponse(
            tracks = tensor_to_bytes(tracks_tensor),
        )

        return response



def run_codigo(request,model,device):
    
    # Check whether a tensor or a video arrived
    if request.video:
        # Write to temp file so decord can open it
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext("temp.mp4")[1] or ".mp4") as tmp:
            tmp.write(request.video)
            tmp.flush()

            # Create VideoReader
            vr = decord.VideoReader(tmp.name)
            frames = vr.get_batch(range(len(vr))).asnumpy()  # (num_frames, H, W, 3)

            # Go to BTCHW
            frames = torch.tensor(np.transpose(frames, (0, 3, 1, 2))).unsqueeze(0).float().to(device) / 255.0  # (1, num_frames, 3, H, W)

            if not request.query_points:
                # use grid_size
                pred_tracks, pred_visibility = model(frames, grid_size=request.grid_size) 
            elif request.query_points:
                # use query_points
                query_points = torch.load(io.BytesIO(request.query_points))
                pred_tracks, pred_visibility = model(frames, query_points=query_points)
            else:
                raise ValueError("Either query_points or grid_size must be provided")
    
    # pred_tracks.shape = (1, num_points, num_frames, 2)
    return pred_tracks, pred_visibility


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
                         options= [('grpc.max_send_message_length', 512 * 1024 * 1024), 
                                   ('grpc.max_receive_message_length', 512 * 1024 * 1024)])
    cotracker_pb2_grpc.add_CoTrackerServiceServicer_to_server(
        CoTrackerService(), server)

    # Add reflection
    service_names = (
        cotracker_pb2.DESCRIPTOR.services_by_name['CoTrackerService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
