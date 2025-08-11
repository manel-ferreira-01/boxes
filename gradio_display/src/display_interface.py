import gradio as gr
import numpy as np
import cv2
from scipy.io import savemat, loadmat
import os
import time

class GradioDisplay:
    def __init__(self,tmp_data_folder="/tmp"):
        # self.(gradio) image_input, label_input,image_outpu,label_output
        self.image = None
        self.label = "User label"
        self.input_data_file=tmp_data_folder+"/in.mat"
        self.output_data_file=tmp_data_folder+"/out.mat"
        self.interface = self._create_interface()

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
                fn=self._update_display,
                inputs=[self.image_input,self.label_input],
                outputs=[self.image_output, self.label_output]
            )
        return demo

    def _update_display(self,img,label):
#if user input an image and clicked on button, store data to be sent by grpc
# and wait for result of processing
        print("update_display: vai guardar no matlab")
        #if there is an input image, process and wait for the answer
        savemat(self.input_data_file,{"img":img,"label":label})
        while True:
            if os.path.exists(self.output_data_file):# Need to lock while loading
               aux=loadmat(self.output_data_file)
               os.remove(self.output_data_file)
               self.image=aux["img"]
               self.label=aux["label"]
               return aux["img"],aux["label"]
            else:
                time.sleep(1)

    def update(self, image_bytes: bytes, label: str):
        try:
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            savemat(self.output_data_file,{"img":image_np,"label":label})
            self.image = image_np
            self.label = label
        except Exception as e:
            print(f"Error in acquire: {e}")
            image_np = np.full((500, 500, 3),255, dtype = np.uint8)
            label="Error"
            savemat(self.output_data_file,{"img":image_np,"label":label})
            self.image = image_np            
            self.label = label

    def launch(self,share=True, server_name="0.0.0.0", server_port=7860):
        self.interface.launch(share=share, server_name=server_name, server_port=server_port)
#        self.interface.launch(share=share, server_name=server_name, server_port=server_port, auth=("jpc", "luz"))

