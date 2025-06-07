import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import simplebox_pb2
import simplebox_pb2_grpc
import inspect
import os
import time
    
import io
from scipy.io import loadmat, savemat
import numpy as np

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ServiceImpl(simplebox_pb2_grpc.SimpleBoxServiceServicer):

    def __init__(self):
        """
        Args:
            
          Loads VGGT model 
        """
        

    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    self._model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)        



    def process(self, request: simplebox_pb2.matfile, context):
        """
        matfile is the matlab file with input data

        Args:
            request: The ImageAndFeatures request to process
            context: Context of the gRPC call

        Returns:
            The Image with the applied function
        features={'kp','desc'}
        """
        datain = request.data

        ret_file= run_codigo(datain, self._model)
        return simplebox_pb2.matfile(data=ret_file)


def run_codigo(datafile,model):
    """
    Reads all variables from a MATLAB .mat file given a file pointer,
    
    Parameters:
    file_pointer (file-like object): Opened .mat file in binary read mode.

    Returns:
    new_file_pointer (io.BytesIO): In-memory file-like object containing the cloned .mat file.
    """
    
    # SPECIFIC CODE STARTS HERE

    #Load the mat file using scipy.io.loadmat
    mat_data=loadmat(io.BytesIO(datafile))

        
    # Extract and flatten the imgdata array
    imgdata = mat_data['imgdata'].squeeze()

    # List to hold in-memory files (as BytesIO) and their associated filenames
    file_buffers = []
    filenames = []

    # Create in-memory files
    for i, data in enumerate(imgdata):
        filename = f'image_{i+1}.jpg'
        buffer = io.BytesIO()
        buffer.write(data.flatten().tobytes())  # Write binary content to buffer
        buffer.seek(0)  # Reset pointer to beginning for future reads
        file_buffers.append(buffer)
        filenames.append(filename)

    images = load_and_preprocess_images(filenames).to(device)
   # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

    p={}
    for k,v in predictions.items():
        p[k]=v.cpu()

    torch.cuda.empty_cache()
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
    simplebox_pb2_grpc.add_SimpleBoxServiceServicer_to_server(
        ServiceImpl(), server)

    # Add reflection
    service_names = (
        simplebox_pb2.DESCRIPTOR.services_by_name['SimpleBoxService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
