import gradio as gr
import numpy as np
import cv2
from scipy.io import savemat, loadmat
import os
import time
import logging
import json
import threading



class GradioDisplay:
    def __init__(self,tmp_data_folder="/tmp",lockfile=None):
        # self.(gradio) image_input, label_input,image_outpu,label_output
        self.image = None
        self.label = "User label"
        self.input_data_file=tmp_data_folder+"/in.mat"
        self.output_data_file=tmp_data_folder+"/out.mat"
        self.lock=lockfile
        self.interface = self._create_interface()
        if os.path.exists(self.input_data_file):
            os.remove(self.input_data_file)
        if os.path.exists(self.output_data_file):
            os.remove(self.output_data_file)
        
            

    def _create_interface(self):

        with gr.Blocks() as demo:
            gr.Markdown("## gRPC Image Display")
            with gr.Row():
                with gr.Column():
                    self.image_input = gr.Image(label="Input Image",type="numpy",interactive=True)
                    self.label_input = gr.Textbox(label="Input,Labels")
                    refresh_btn = gr.Button("🔄 Run Pipeline ", elem_id="refresh-btn")
                with gr.Column():
                    self.image_output = gr.Image(label="Received Image", interactive=False)
                    self.label_output = gr.Textbox(label="Label", interactive=False)
            #   Metodos- Button click
            refresh_btn.click(
                fn=self._update_acquire,
                inputs=[self.image_input,self.label_input],
                outputs=[self.image_output, self.label_output]
            )
        return demo

    def _update_acquire(self,img,label):
    #if user input an image and clicked on button, store data to be sent by grpc
    # and wait for result of processing
        if img is None:
            return None, "No image"
        #if there is an input image, process and wait for the answer
        try:
#            print("UPDATEACQUIRE: vai ao lock")
            with self.lock:
                savemat(self.input_data_file,{"img":img,"label":label})
 #           print("UPDATEACQUIRE: saiu do lock")
        except Exception as e:
            logging.error(f"Error in update_display SAVEMAT: {e}")
            return None, f"Error in update_display SAVEMAT: {e}"

        while True:
            try:
                if os.path.exists(self.output_data_file):# Need to lock while loading
                    with self.lock:
                        aux=loadmat(self.output_data_file)
                        os.remove(self.output_data_file)
#                    print("UPDATEACQUIRE: saiu do segunbdo lock")
                    return aux["img"],json.loads(aux["label"].tobytes())
                else:
                    time.sleep(.1)
            except Exception as e:
                logging.error(f"UPDATE_ACQUIRE: Error during loadmat: {e}")
        #------------- print ---------
#                print(f"UPDATE_ACQUIRE erro {e}")
                time.sleep(10)
                return None,"Error in data"

    def update(self, image_bytes: bytes, label: str):
        try:
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
#            print("UPDATE: vai ao lock")
            with self.lock:
                savemat(self.output_data_file,{"img":image_np,"label":label})
#            print("UPDATE: Saiu do  lock")
        except Exception as e:
            logging.error(f"Error in update: {e}")
            image_np = np.full((500, 500, 3),255, dtype = np.uint8)
            label="Error"
            with self.lock:
                savemat(self.output_data_file,{"img":image_np,"label":label})

    def launch(self,share=True, server_name="0.0.0.0", server_port=7860):
        self.interface.launch(share=share, server_name=server_name, server_port=server_port)
#        self.interface.launch(share=share, server_name=server_name, server_port=server_port, auth=("jpc", "luz"))

