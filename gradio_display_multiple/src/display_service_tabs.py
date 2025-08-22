#from tkinter import LabelFrame
import grpc
from concurrent import futures
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import display_pb2
import display_pb2_grpc
from display_interface_tabs import GradioDisplay
from scipy.io import loadmat, savemat
import numpy as np
import cv2
import json
import datetime
import os
import time
import threading
import pickle

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
            print("Tamanho do pickle")
            print(os.path.getsize(self.gradio_display.input_data_file))
            try:
                with lock:
                    print("Há DADOS")
                    with open(self.gradio_display.input_data_file, 'rb') as f:
                        gradio_data=pickle.load(f)  
                    os.remove(self.gradio_display.input_data_file)
                # generate a list with one single image
                if gradio_data["command"]=="single":
                    img=gradio_data['gradio'][0][0];
                    print(img.shape)
                    image_bytes=[cv2.imencode('.jpg', img)[1].tobytes() ]
                     #-----generate a list of encoded images from a gallery
                elif gradio_data["command"]=="sequence":
                    gg=gradio_data["gradio"][0] #a list of tuples [(im,caption)]                
                    image_bytes = [cv2.imencode('.jpg', img)[1].tobytes() for img in [im for im in [ggg[0]  for ggg in gg]]]

                #--- Add annotations + counting, datetime
                self.input_count=self.input_count+1
                annotations={ "command":gradio_data["command"], 
                              "user":gradio_data["gradio"][1],
                              "input_count":self.input_count,
                              "timestamp":datetime.datetime.now().isoformat(),
                              "yoloconfig":"vai aqui a configuraçao"
                            }
                label= json.dumps([{"aispgradio":annotations}])
                
                return display_pb2.AcquireResponse(label=label, image=image_bytes)
            except Exception as e:
                logging.error(f"Error in acquire: {e}")
                time.sleep(0.1)
                annotations={"erroracquire":f"Error in acquire: {e}"}
        else:
            time.sleep(2)
            annotations= {"empty":"empty"}
            

        label= json.dumps([{"aispgradio":annotations}])
        _,tmp=cv2.imencode('.jpg',np.zeros((2,2,3),dtype='uint8'))

        return display_pb2.AcquireResponse(label=label, image=[tmp.tobytes()])


    def display(self, request, context):
        label=json.loads(request.label) 
#        print(label)
        try:#--- parse the label info
            for l in label:
#                print("--LABEL---")
#                print(l)
                if type(l) is dict:
                    if "aispgradio" in l:
                        if "empty" in l['aispgradio']:
                            return display_pb2.DisplayResponse()
                        elif "single" in l["aispgradio"]["command"]:
                            print("--chegou imagem single---")
                            self.gradio_display.update(request.image[0], request.label)
                        elif "sequence" in l["aispgradio"]["command"] :
                            print("---chegou sequencia ------ num images: {len(request.image)}")
                            image_np = [cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR) for img in request.image]
                            with lock:
                                with open(self.gradio_display.output_data_file, 'wb') as f:
                                    pickle.dump([image_np,request.label],f)
                        else:
                            print("Tem AISPGRADIO MAS NAO APANHOU KEYWORD NENHUMA")
                            print(l["aispgradio"])
        
        except Exception as e:
            print("Erro no DISPLAY ")
            print(e)
        print("------------End Display -------------------------")    
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



