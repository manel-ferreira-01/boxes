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

# add vggt to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/vggt')
print(sys.path)

from PIL import Image
import torch.nn.functional as F

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
import os
from torchvision import transforms as TF

_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def preprocess_images_batch(image_tensor_batch, mode="crop"):
    """
    Preprocess a batch of images for model input.

    Assumes tensor is in (S, C, H, W) format with values in [0, 1].

    Args:
        image_tensor_batch (torch.Tensor): Batch of images (S, C, H, W)
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (S, 3, H, W)

    Raises:
        ValueError: If the input is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - When mode="crop": Ensures width=518px while maintaining aspect ratio,
          and height is center-cropped if larger than 518px
        - When mode="pad": Ensures the largest dimension is 518px while maintaining aspect ratio,
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14
    """
    if image_tensor_batch.ndim != 4:
        raise ValueError("Input must be a 4D tensor (S, C, H, W)")

    if image_tensor_batch.size(0) == 0:
        raise ValueError("At least 1 image is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    target_size = 518
    shapes = set()
    processed_images = []

    for img in image_tensor_batch:
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError("Each image must be in (3, H, W) format")

        _, height, width = img.shape

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # mode == "crop"
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize
        img = F.interpolate(
            img.unsqueeze(0),                # (1, C, H, W)
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)                         # back to (C, H, W)

        # Center crop height if it's larger than target_size (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to square target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        processed_images.append(img)

    # If different shapes, pad to max shape
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        padded_images = []
        for img in processed_images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        processed_images = padded_images

    return torch.stack(processed_images)

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t.cpu(), buf)
    return buf.getvalue()

class VGGTService(vggt_pb2_grpc.VGGTServiceServicer):

    def __init__(self):
        """
        Args:
          Loads VGGT model 
        """
        

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.
        self._model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)     
        logging.info(f'Model loaded on device: {device}')


    def Forward(self, request, context):
        """
        matfile is the matlab file with input data

        Args:
            request: The ImageAndFeatures request to process
            context: Context of the gRPC call

        Returns:
            The Image with the applied function
        features={'kp','desc'}
        """
        #datain = request.images

        ret_file= run_codigo(request, self._model,self._device)

        response = vggt_pb2.VGGTResponse(
            pose_enc=tensor_to_bytes(ret_file["pose_enc"]),
            depth=tensor_to_bytes(ret_file["depth"]),
            depth_conf=tensor_to_bytes(ret_file["depth_conf"]),
            world_points=tensor_to_bytes(ret_file["world_points"]),
            world_points_conf=tensor_to_bytes(ret_file["world_points_conf"]),
        )

        # If your ret_file sometimes includes these optional outputs:
        if "track" in ret_file:
            response.track = tensor_to_bytes(ret_file["track"])
        if "vis" in ret_file:
            response.vis = tensor_to_bytes(ret_file["vis"])
        if "conf" in ret_file:
            response.conf = tensor_to_bytes(ret_file["conf"])

        return response



def run_codigo(datafile,model,device):
    """
    Reads all variables from a MATLAB .mat file given a file pointer,
    
    Parameters:
    file_pointer (file-like object): Opened .mat file in binary read mode.

    Returns:
    new_file_pointer (io.BytesIO): In-memory file-like object containing the cloned .mat file.
    """
    
    # SPECIFIC CODE STARTS HERE

    received_images = []
    for image_bytes in datafile.images:
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream).convert("RGB")
        img_np = np.array(img)
        received_images.append(img_np)

    images = torch.tensor(np.stack(received_images)).permute(0,3,1,2).to(device)
    images = preprocess_images_batch(images.float() / 255)

    query_points = None
    #if hasattr(datafile, 'query_points') and datafile.query_points:
    #    query_points = tensor_from_proto(datafile.query_points).to(device)

    # bfloat16 is supported on Ampere+
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images, query_points=query_points)

    # Drop unnecessary intermediate outputs
    predictions.pop("pose_enc_list", None)

    # Move everything to CPU for serialization
    return {k: v.cpu() for k, v in predictions.items()}


    # SPECIFIC CODE ENDS HERE

    f=io.BytesIO()
    # WRITE RETURNING DATA the predictions dictionary
    savemat(f,p)
    return f.getvalue()

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
