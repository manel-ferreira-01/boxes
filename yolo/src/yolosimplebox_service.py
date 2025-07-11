import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import yolosimplebox_pb2
import yolosimplebox_pb2_grpc
import inspect
import os
import time
import numpy as np
import torch
    
import io
from scipy.io import loadmat, savemat
from ultralytics import YOLO

# VERIFY THE PORT NUMBER 
_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class ServiceImpl(yolosimplebox_pb2_grpc.SimpleBoxServiceServicer):

    def __init__(self):
        """
        Args:
            calling_function: the function that should be called
                              when a new request is received

                              the signature of the function should be:

                              (image: bytes) -> bytes

                              as described in the process method

        """
        self.model = YOLO("yolo11n.pt")

 #       https://docs.ultralytics.com/modes/predict/#inference-arguments
        


    def detect(self, request: yolosimplebox_pb2.matfile, context):
        """
        matfile is the matlab file with input data

        Args:
            request: The Images request to process
            context: Context of the gRPC call

        Returns:
            a matlab file with the results object from yolo prediction
        """
        datain = request.data

        ret_file= run_codigo(datain,self.model)
        return yolosimplebox_pb2.matfile(data=ret_file)

# THIS IS WHERE YOUR CODE MUST BE INSERTED

def to_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    elif isinstance(value, (list, tuple)):
        return value
    else:
        return None

def extract_array_attributes(obj, prefix=''):
    array_types = (list, tuple, np.ndarray, torch.Tensor)
    attr_dict = {}
    
    for attr in dir(obj):
        if attr.startswith("_"):
            continue  # Skip private/internal attributes
        try:
            value = getattr(obj, attr)
            # Check for nested objects like boxes, masks, keypoints
            if hasattr(value, '__dict__') and not isinstance(value, (np.ndarray, torch.Tensor)):
                nested = extract_array_attributes(value, prefix=prefix + attr + '.')
                attr_dict.update(nested)
            elif isinstance(value, torch.Tensor):
                attr_dict[prefix + attr] = value.detach().cpu().numpy()
            elif isinstance(value, array_types):
                attr_dict[prefix + attr] = value
        except Exception:
            continue
    return attr_dict


def run_codigo(datafile,model):
    """
    Reads all variables from a MATLAB .mat file given a file pointer,
    
    Parameters:
    file_pointer (file-like object): Opened .mat file in binary read mode.

    Returns:
    new_file_pointer (io.BytesIO): In-memory file-like object containing the cloned .mat file.
    """
    #Load the mat file using scipy.io.loadmat
    mat_data=loadmat(io.BytesIO(datafile))

    # SPECIFIC CODE STARTS HERE
    im=mat_data["im"]
    results=model(im)

    all_arrays = []

    for i, result in enumerate(results):
        #arr_dict = extract_array_attributes(result, prefix=f'result[{i}].')
        arr_dict=result.summary()
        all_arrays.append(arr_dict)


    # SPECIFIC CODE ENDS HERE

    f=io.BytesIO()
    # WRITE RETURNING DATA
    savemat(f,{"results":all_arrays})
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
    yolosimplebox_pb2_grpc.add_SimpleBoxServiceServicer_to_server(
        ServiceImpl(), server)

    # Add reflection
    service_names = (
        yolosimplebox_pb2.DESCRIPTOR.services_by_name['SimpleBoxService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
