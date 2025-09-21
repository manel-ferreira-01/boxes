import grpc
from concurrent import futures
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
from display_interface import GradioDisplay
import numpy as np
import cv2
import json
import datetime
import os
import time
import threading
import pickle

import sys
sys.path.append("./protos")
import pipeline_pb2 as display_pb2
import pipeline_pb2_grpc as display_pb2_grpc
from aux import wrap_value, unwrap_value

lock = threading.Lock()
# --- Configuration ---

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class DisplayService(display_pb2_grpc.DisplayServiceServicer):
    def __init__(self, gradio_display: GradioDisplay):
        self.gradio_display = gradio_display
        self.input_count=0

    def acquire(self, request, context):       # while True:        for i in range(3):
        # Read in.mat if exists and returns grpc message
        if os.path.exists(self.gradio_display.input_data_file):# Need to lock while loading
            try:
                with lock:
                    with open(self.gradio_display.input_data_file, 'rb') as f:
                        gradio_data=pickle.load(f)  
                    os.remove(self.gradio_display.input_data_file)
                # generate a list with one single image
                if gradio_data["command"]=="single":
                    img=gradio_data['gradio'][0][0]
                    image_bytes=[cv2.imencode('.jpg', img)[1].tobytes() ]
                     #-----generate a list of encoded images from a gallery

                elif gradio_data["command"]=="detectsequence" or gradio_data["command"]=="tracksequence":
                    gg=gradio_data["gradio"][0] #a list of tuples [(im,caption)]                
                    image_bytes = [cv2.imencode('.jpg', img)[1].tobytes() for img in [im for im in [ggg[0]  for ggg in gg]]]

                elif gradio_data["command"]=="3d_infer":
                    logging.info("3D INFER - acquire")
                    gg=gradio_data["gradio"][0] #a list of tuples [(im,caption)]                
                    image_bytes = [cv2.imencode('.jpg', img)[1].tobytes() for img in [im for im in [ggg[0]  for ggg in gg]]]

                else: # Update for the case of now labels
                    logging.error(f"No command in the json string")
                    
                #--- Add annotations + counting, datetime
                self.input_count=self.input_count+1
                annotations={ "command":gradio_data["command"], 
                              "user":gradio_data["gradio"][1], # supposedely it is "user input"
                              "input_count":self.input_count,
                              "timestamp":datetime.datetime.now().isoformat(),
                              "parameters":gradio_data["parameters"] #optional parameters
                            }
            except Exception as e:
                logging.error(f"Error in acquire: {e}")
                time.sleep(0.1)
                annotations={"erroracquire":f"Error in acquire: {e}"}
        else:
            time.sleep(2)
            annotations= {"empty":"empty"}
            _,tmp=cv2.imencode('.jpg',np.zeros((2,2,3),dtype='uint8')) #zero image just in case
            image_bytes= [tmp.tobytes()]
            logging.info(f"DISPLAY : No data file {self.gradio_display.input_data_file} found, sending empty response")

        out_json = json.dumps({"aispgradio":annotations})
        images = image_bytes

        return display_pb2.Envelope(config_json=out_json,
                                    data={"images": wrap_value(images)} ) # list of bytes


    def display(self, request, context):

        logging.info("DISPLAY : New display request")
        logging.info(request.config_json) 
        label=json.loads(request.config_json) 
        logging.info(label)
        
        for l in label:
            if "aispgradio" == l:
                if "empty" in label['aispgradio'].keys():
                    return display_pb2.Envelope()
                
                #elif "single" in l["aispgradio"]["command"]:
                #    print("--chegou imagem single---")
                #    self.gradio_display.update(request.image[0], request.label)
                #elif "detectsequence" in l["aispgradio"]["command"] :
                #    image_np = [cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR) for img in request.image]
                #    with lock:
                #        with open(self.gradio_display.output_data_file, 'wb') as f:
                #            pickle.dump([image_np,request.label],f)
                #            
                #elif "tracksequence" in l["aispgradio"]["command"] :
                #    image_np = [cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR) for img in request.image]
                #    with lock:
                #        with open(self.gradio_display.output_data_file, 'wb') as f:
                #            pickle.dump([image_np,request.label],f)

                elif "3d_infer" == label["aispgradio"]["command"]:
                    logging.info("3D INFER - display")
                    glb_file = (unwrap_value(request.data["glb_file"])) if unwrap_value(request.data["glb_file"]) else None
                    with lock:
                        with open(self.gradio_display.output_data_file, 'wb') as f:
                            pickle.dump([glb_file,request.config_json],f)
                            logging.info("saved the glb_file")
                            
                else:
                    tmp=label["aispgradio"]
                    logging.error(f"Tem AISPGRADIO MAS NAO APANHOU KEYWORD NENHUMA {tmp}")
        
           
        print("------------End Display -------------------------")    
        logging.info("DISPLAY : End display")
        return display_pb2.Envelope()

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
        options= [('grpc.max_send_message_length', -1), 
                  ('grpc.max_receive_message_length', -1)])
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



