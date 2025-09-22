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
import sys
sys.path.append("./protos")
import pipeline_pb2 as vggt_pb2
import pipeline_pb2_grpc as vggt_pb2_grpc
from aux import wrap_value, unwrap_value


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


class PipelineService(vggt_pb2_grpc.PipelineServiceServicer):

    def __init__(self):
        # Always load to CPU first
        self._model = VGGT()
        self._model.load_state_dict(
            torch.load("./vggt-1b.pt", map_location="cpu")
        )
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
                if idle_time > IDLE_TIMEOUT and self._device == "cuda:0":
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"

    def move_to_gpu(self):
        with self._lock:
            self._last_request_time = time.time()
            if self._device == "cpu" and torch.cuda.is_available():
                logging.info("Moving model to GPU")
                self._model.to("cuda:0")
                self._device = "cuda:0"

    def Process(self, request, context):

        results = {}
        if request.config_json:
            config_json = json.loads(request.config_json)
            logging.info(config_json)
            for entry in config_json:
                if entry == "aispgradio":
                    if "empty" in config_json[entry].keys():
                        logging.info("Empty request received, returning empty response")
                        return vggt_pb2.Envelope(config_json= json.dumps({'aispgradio': {'empty': 'empty'}}))
                    elif "command" in config_json[entry]:
                        if "3d_infer" in config_json[entry]["command"]:
                            logging.info("3D inference request received")
                            # Run inference
                            self.move_to_gpu()
                            results, glb_file = run_codigo(request, self._model, self._device)
                            logging.info("3D inference completed")
                        else:
                            logging.error(f"Unknown command {config_json[entry]['command']}, returning empty response")
                            return vggt_pb2.Envelope(config_json= json.dumps({'aispgradio': {'empty': 'empty'}}))
        else:
            logging.info("No config_json provided, returning empty response")
            return vggt_pb2.Envelope(config_json= json.dumps({'aispgradio': {'empty': 'empty'}}))
        
        # if there is a valid response, serialize tensors to bytes
        import zlib
        response = vggt_pb2.Envelope(
            config_json=json.dumps({'aispgradio': {'command': '3d_infer'}}),
            data={"world_points": wrap_value(tensor_to_bytes(results["world_points"])),
                "world_points_conf": wrap_value(tensor_to_bytes(results["world_points_conf"])),
                "depth": wrap_value(tensor_to_bytes(results["depth"])),
                "depth_conf": wrap_value(tensor_to_bytes(results["depth_conf"])),
                "extrinsic": wrap_value(tensor_to_bytes(results["extrinsic"])),
                "intrinsic": wrap_value(tensor_to_bytes(results["intrinsic"])),
                "images": wrap_value(tensor_to_bytes(torch.tensor(results["images"]))),
                "glb_file" : wrap_value(glb_file)} # already bytes
        )

        #print the size of glb_file in bytes
        if 'glb_file' in response.data:
            glb_size = len(unwrap_value(response.data['glb_file']))
            logging.info(f"Size of glb_file: {glb_size / (1024 * 1024):.2f} MB")
            logging.info(f"Size of glb_file (compressed): {len(zlib.compress(unwrap_value(response.data['glb_file'])))/(1024*1024):.2f} MB")

        logging.info("Returning response from Process")
        return response

def run_codigo(request,model,device):

    received_images = []
    for image_bytes in unwrap_value(request.data["images"]):
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
    predictions["extrinsic"] = extrinsic.squeeze() # TODO: ADD THIS TO THE PROTO
    predictions["intrinsic"] = intrinsic.squeeze()

    # Drop unnecessary intermediate outputs
    predictions.pop("pose_enc_list", None)
    predictions = {k: v.cpu() for k, v in predictions.items()}

    #add images to output
    predictions["images"] = images.cpu().numpy()

    #extract conf threshold from request
    json_config = json.loads(request.config_json)
    if json_config["aispgradio"]["parameters"] and "conf_threshold" in json_config["aispgradio"]["parameters"]:
        conf_thres = json_config["aispgradio"]["parameters"]["conf_threshold"]
    else:
        conf_thres = 30  # default value
    logging.info(f"Using confidence threshold: {conf_thres}")

    # create a file like object to export the glb file
    glb_scene = predictions_to_glb(predictions, conf_thres=conf_thres)
    b = glb_scene.export(file_type="glb")

    # Move everything to CPU for serialization
    return predictions, b



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
    vggt_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    # Add reflection
    service_names = (
        vggt_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
