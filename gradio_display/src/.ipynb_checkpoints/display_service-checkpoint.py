#from tkinter import LabelFrame
import grpc
from concurrent import futures
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import display_pb2
import display_pb2_grpc
from display_interface import GradioDisplay
from scipy.io import loadmat, savemat
import numpy as np
import cv2
import json
import datetime
import os
import time
import threading

lock = threading.Lock()
# --- Configuration ---

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class DisplayService(display_pb2_grpc.DisplayServiceServicer):
    def __init__(self, gradio_display: GradioDisplay):
        self.gradio_display = gradio_display
        self.input_count=0;

    def acquire(self, request, context):       # while True:        for i in range(3):
        # Read in.mat if exists and returns grpc message
        if os.path.exists(self.gradio_display.input_data_file):# Need to lock while loading
            try:
#                print("ACQUIRE: Vai fazer o lock")
                with lock:
                    aux=loadmat(self.gradio_display.input_data_file)
                    os.remove(self.gradio_display.input_data_file)
#                print("ACQUIRE: saiu lock")
                _, image_bytes = cv2.imencode('.jpg', aux["img"])
              #Add annotations: counting, datetime
                self.input_count=self.input_count+1
                annotations={"user":np.array2string(aux["label"]),"input_count":self.input_count,
                             "timestamp":datetime.datetime.now().isoformat()}
                label= json.dumps({"aispgradio":annotations})
                return display_pb2.AcquireResponse(label=label, image=image_bytes.tobytes())
            except Exception as e:
                logging.error(f"Error in acquire: {e}")
                time.sleep(0.1)
                annotations={"erroracquire":f"Error in acquire: {e}"}
        else:
            time.sleep(2)
            annotations= {"empty":""}
            

        label= json.dumps({"aispgradio":annotations})  
        _,tmp=cv2.imencode('.jpg',np.zeros((2,2,3),dtype='uint8'))
        return display_pb2.AcquireResponse(label=label, image=tmp.tobytes())


    def display(self, request, context):
        label=json.loads(request.label)
        
        for l in label:
            if type(l) is dict:
                if "aispgradio" in l.keys():
                    if "empty" in l['aispgradio']:
                        return display_pb2.DisplayResponse()
        
        self.gradio_display.update(request.image, request.label)   
        return display_pb2.DisplayResponse()

# --- gRPC Server Launchers ---

def run_server(server,gradio_display):

    server_port = int(os.getenv('PORT', _PORT_DEFAULT))
    target = f'[::]:{server_port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'''Server started at {target}''')
    gradio_display.launch()
    logging.info(f'''Launched GRADIO ----- RUN forever ''')    
    server.wait_for_termination()

if __name__ == "__main__":

    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)
    #Create Server and add service
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options= [('grpc.max_send_message_length', 512 * 1024 * 1024), 
                  ('grpc.max_receive_message_length', 512 * 1024 * 1024)])
#__ Create Gradio services __ 
    gradio_display = GradioDisplay(tmp_data_folder="/tmp",lockfile=lock)

# -- Add service 
    display_pb2_grpc.add_DisplayServiceServicer_to_server(DisplayService(gradio_display), server)

    # Add reflection
    service_names = (
        display_pb2.DESCRIPTOR.services_by_name['DisplayService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)
    print("Running server GRPC")
    run_server(server,gradio_display)



