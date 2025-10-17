import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
    
import io
import numpy as np
import json
import torch

# add vggt to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lang-segment-anything')
print(sys.path)

from PIL import Image
import pickle

from importlib.machinery import SourceFileLoader
import sys
sys.path.append("./protos")
import pipeline_pb2 as lang_sam_pb2
import pipeline_pb2_grpc as lang_sam_grpc
from aux import wrap_value, unwrap_value

import threading
_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
IDLE_TIMEOUT = 60  # seconds

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/lang-segment-anything')
from lang_sam import LangSAM

class PipelineService(lang_sam_grpc.PipelineServiceServicer):
    def __init__(self):
        # Always load to CPU first
        self._model = LangSAM(sam_type="sam2.1_hiera_small",device="cpu")
        self._device = "cpu"
        print("Model loaded on CPU")
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
                new_model = LangSAM(device=target)
                # Replace old one
                del self._model
                torch.cuda.empty_cache()
                self._model = new_model
                self._device = target
            except Exception as e:
                logging.exception(f"Failed to move model to {target}: {e}")
            return self._device

    def Process(self, request, context):
        """Perform text-guided segmentation on image(s)."""
        try:
            # --- Validate request ---
            if not request.config_json:
                raise ValueError("Missing config_json in request")

            try:
                config = json.loads(request.config_json)
            except json.JSONDecodeError:
                raise ValueError("config_json is not valid JSON")


            params = config.get("parameters", {}) or {}
            requested_device = params.get("device", None)
            if requested_device:
                self.set_device(requested_device)

            # --- Extract image(s) ---º
            img_list = unwrap_value(request.data.get("images", []))
            print(img_list)
            if not img_list:
                raise ValueError("No images provided in request.data['images']")
            
            
            # --- Run inference ---
            results = self.infer_lang_sam(img_list, config)

            # --- Build response ---
            return lang_sam_pb2.Envelope(
                data={"results": wrap_value(pickle.dumps(results))},
                config_json=json.dumps({"aispgradio": {"status": "ok"}})
            )

        except Exception as e:
            #logging.error(f"[InferLangSAM] {e}")
            return lang_sam_pb2.Envelope(
                config_json=json.dumps({"aispgradio": {"error": str(e)}})
            )



    def infer_lang_sam(self, request):

        # read the images
        received_images = []
        for image_bytes in unwrap_value(request.data["images"]):
            image_stream = io.BytesIO(image_bytes)
            img = Image.open(image_stream).convert("RGB")
            #img_np = np.array(img)
            received_images.append(img)

        #read the text prompts
        text_prompts = json.loads(request.config_json)["aispgradio"].get("text_prompt", [])
        out_list = []
        for image in received_images:
            output = self._model.predict([image], text_prompts)
            out_list.append(output[0])

        return out_list


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
    lang_sam_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    # Add reflection
    service_names = (
        lang_sam_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
