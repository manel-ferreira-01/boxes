import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
    
import io
import numpy as np

import torch

# add vggt to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/vggt')
print(sys.path)

from PIL import Image
import pickle

from importlib.machinery import SourceFileLoader
vggt_pb2 = SourceFileLoader(
    "vggt_pb2",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../protos/vggt_pb2.py")
).load_module()
vggt_pb2_grpc = SourceFileLoader(
    "vggt_pb2_grpc",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../protos/vggt_pb2_grpc.py")
).load_module()

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.visual_util import predictions_to_glb
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import os
from torchvision import transforms as TF
from utils.preprocess import preprocess_images_batch

_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
IDLE_TIMEOUT = 60  # seconds

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.cpu(), buf)
    return buf.getvalue()

import threading
import json


class VGGTService(vggt_pb2_grpc.VGGTServiceServicer):

    def __init__(self):
        # Always load to CPU first
        self._model = VGGT.from_pretrained("facebook/VGGT-1B").to("cpu")
        self._device = "cpu"
        logging.info("Model loaded on CPU")

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
                if idle_time > IDLE_TIMEOUT and self._device == "cuda:1":
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"

    def Forward(self, request, context):

        with self._lock:
            self._last_request_time = time.time()
            # If idle watchdog moved it back to CPU, restore to GPU
            if self._device == "cpu" and torch.cuda.is_available():
                logging.info("Request received: moving model to GPU")
                self._model.to("cuda:1")
                self._device = "cuda:1"

        results = {}
        if request.config_json:
            config_json = json.loads(request.config_json)
            print(config_json)
            for entry in config_json:
                if entry == "aispgradio":
                    if "empty" in config_json["aispgradio"].keys():
                        logging.info("Empty request received, returning empty response")
                        return vggt_pb2.VGGTResponse()
                    elif "command" in config_json["aispgradio"]:
                        if "3d_infer" in config_json["aispgradio"]["command"]:
                            logging.info("3D inference request received")
                            # Run inference
                            results, glb_file = run_codigo(request, self._model, self._device)
        else:
            logging.info("No config_json provided, returning empty response")
            return vggt_pb2.VGGTResponse()
        
        # if there is a valid response, serialize tensors to bytes
        response = vggt_pb2.VGGTResponse(
            pose_enc=tensor_to_bytes(results["pose_enc"]),
            depth=tensor_to_bytes(results["depth"]),
            depth_conf=tensor_to_bytes(results["depth_conf"]),
            world_points=tensor_to_bytes(results["world_points"]),
            world_points_conf=tensor_to_bytes(results["world_points_conf"]),
            images=tensor_to_bytes(torch.tensor(results["images"])),
            vis=pickle.dumps(glb_file)
        )


        return response

def run_codigo(datafile,model,device):

    received_images = []
    for image_bytes in datafile.images:
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream).convert("RGB")
        img_np = np.array(img)
        received_images.append(img_np)

    images = torch.tensor(np.stack(received_images)).permute(0,3,1,2).to(device)
    images = preprocess_images_batch(images.float() / 255)

    query_points = None

    # bfloat16 is supported on Ampere+
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images, query_points=query_points)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic # TODO: ADD THIS TO THE PROTO
    predictions["intrinsic"] = intrinsic

    # Drop unnecessary intermediate outputs
    predictions.pop("pose_enc_list", None)
    predictions = {k: v.cpu() for k, v in predictions.items()}

    #add images to output
    predictions["images"] = images.cpu().numpy()

    glb_file = predictions_to_glb(predictions)

    # Move everything to CPU for serialization
    return predictions, glb_file



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
    vggt_pb2_grpc.add_VGGTServiceServicer_to_server(
        VGGTService(), server)

    # Add reflection
    service_names = (
        vggt_pb2.DESCRIPTOR.services_by_name['VGGTService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
