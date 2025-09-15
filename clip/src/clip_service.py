import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
    
import io
import os

import torch
import clip
import torch.nn.functional as F
from PIL import Image



from importlib.machinery import SourceFileLoader

clip_pb2 = SourceFileLoader(
    "clip_pb2",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "./clip_pb2.py")
).load_module()
clip_pb2_grpc = SourceFileLoader(
    "clip_pb2_grpc",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "./clip_pb2_grpc.py")
).load_module()


_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.cpu(), buf)
    return buf.getvalue()

import threading

IDLE_TIMEOUT = 60  # seconds (1 min)

class CLIPService(clip_pb2_grpc.CLIPServiceServicer):

    def __init__(self):
        # Always load to CPU first
        self._model, self._preprocess = clip.load("ViT-B/32", device="cpu",
                                                  download_root="./model")
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
                if idle_time > IDLE_TIMEOUT and self._device == "cuda":
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
                self._model.to("cuda")
                self._device = "cuda"

        # Run inference
        image_features, text_features, logits_per_image = run_codigo(request, self._model, self._device, self._preprocess)

        response = clip_pb2.CLIPResponse(
            image_emb=tensor_to_bytes(image_features),
            text_emb=tensor_to_bytes(text_features),
            similarity=tensor_to_bytes(torch.tensor(logits_per_image).detach())
        )

        return response



def run_codigo(request,model,device, preprocess):

    received_images = []
    for image_bytes in request.images:
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream).convert("RGB")
        received_images.append(img)


    images = torch.stack([preprocess(im) for im in received_images]).to(device)
    text = clip.tokenize(request.texts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(images, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return image_features, text_features, logits_per_image

    

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
    clip_pb2_grpc.add_CLIPServiceServicer_to_server(
        CLIPService(), server)

    # Add reflection
    service_names = (
        clip_pb2.DESCRIPTOR.services_by_name['CLIPService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
