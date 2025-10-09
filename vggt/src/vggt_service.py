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
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            with self._lock:
                idle_time = time.time() - self._last_request_time
                # Move back to CPU only if currently on GPU
                if idle_time > IDLE_TIMEOUT and self._device.startswith("cuda"):
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"

    def set_device(self, target: str):
        with self._lock:
            self._last_request_time = time.time()
            target = target.lower()
            if target.startswith("cuda") and not torch.cuda.is_available():
                logging.warning("CUDA requested but not available. Staying on CPU.")
                return self._device

            if target == self._device:
                return self._device

            try:
                logging.info(f"Reinitializing VGGT model on {target}")
                # Reinstantiate model fresh on target device
                new_model = VGGT().to(target)
                new_model.load_state_dict(
                    self._model.state_dict(), strict=False
                )
                # Replace old one
                del self._model
                torch.cuda.empty_cache()
                self._model = new_model
                self._device = target
            except Exception as e:
                logging.exception(f"Failed to move model to {target}: {e}")
            return self._device


    def Process(self, request, context):

        results = {}
        if request.config_json:
            config_json = json.loads(request.config_json)
            for entry in config_json:
                if entry == "aispgradio":
                    if "empty" in config_json[entry].keys():
                        return vggt_pb2.Envelope(config_json=json.dumps({'aispgradio': {'empty': 'empty'}}))
                    elif "command" in config_json[entry]:
                        if "3d_infer" in config_json[entry]["command"]:

                            # Handle optional device parameter
                            parameters = config_json[entry].get("parameters", {}) or {}
                            requested_device = parameters.get("device")
                            if requested_device:
                                    new_dev = self.set_device(requested_device)

                            logging.info(f"Using device for inference: {new_dev}")
                            logging.info("3D inference request received")
                            results, glb_file = run_codigo(request, self._model, self._device)
                            
                        else:
                            return vggt_pb2.Envelope(config_json=json.dumps({'aispgradio': {'empty': 'empty'}}))
        else:
            return vggt_pb2.Envelope(config_json=json.dumps({'aispgradio': {'empty': 'empty'}}))
        
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

def run_codigo(request, model, device):

    received_images = []
    for image_bytes in unwrap_value(request.data["images"]):
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream).convert("RGB")
        img_np = np.array(img)
        received_images.append(img_np)

    #check if images are all the same shape otherwise raise error
    shapes = [img.shape for img in received_images]
    if len(set(shapes)) != 1:
        raise ValueError(f"All images must have the same shape, but got shapes: {shapes}")
    
    images = torch.tensor(np.stack(received_images)).permute(0,3,1,2).to(device)
    images = preprocess_images_batch(images.float() / 255)

    query_points = None

    use_cuda = device.startswith("cuda")
    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images, query_points=query_points)
    else:
        with torch.no_grad():
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
    params = json_config.get("aispgradio", {}).get("parameters", {}) or {}
    conf_thres = params.get("conf_threshold", 30)
    logging.info(f"Using confidence threshold: {conf_thres}")

    # create a file like object to export the glb file
    glb_scene = predictions_to_glb(predictions, conf_thres=conf_thres)
    b = glb_scene.export(file_type="glb")

    # Move everything to CPU for serialization
    return predictions, b

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
    vggt_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    # Add reflection
    service_names = (
        vggt_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
