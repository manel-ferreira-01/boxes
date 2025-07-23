import queue
import time
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
from concurrent import futures

import acquisition_pb2
import acquisition_pb2_grpc

# Fill the queue with sample data for demo
from PIL import Image
import io
import base64
_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


# This is your global queue
data_acq_queue = queue.Queue(maxsize=3)

# GRPC server implementation
class AcquisitionServiceServicer(acquisition_pb2_grpc.AcquisitionServiceServicer):

    def __init__(self):
        """
        Args:
            
            """
        
    def acquire(self, request, context):

        try:
            label, image_bytes = data_acq_queue.get()
            return acquisition_pb2.AcquireResponse(label=label, image=image_bytes)
        except queue.Empty:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('No data available')
            return acquisition_pb2.AcquireResponse()

        
def image_to_bytes(img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# Add dummy images and labels to the queue
#for i in range(5):
#    img = Image.new('RGB', (100, 100), (i*40, i*20, i*30))
#    data_queue.put(("Label_" + str(i), image_to_bytes(img)))

    
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
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options= [('grpc.max_send_message_length', 512 * 1024 * 1024), 
                                   ('grpc.max_receive_message_length', 512 * 1024 * 1024)])

    acquisition_pb2_grpc.add_AcquisitionServiceServicer_to_server(
        AcquisitionServiceServicer(), server)
    # Add reflection
    service_names = (
        acquisition_pb2.DESCRIPTOR.services_by_name['AcquisitionService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)

