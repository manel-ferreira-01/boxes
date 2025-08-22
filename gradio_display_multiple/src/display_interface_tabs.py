import gradio as gr
import numpy as np
import cv2
from scipy.io import savemat, loadmat
import os
import time
import logging
import json
import threading
import pickle


class GradioDisplay:
    def __init__(self,tmp_data_folder="/dev/shm",lockfile=None):
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
        
            
#--------  Create Gradio Interface --------
    
    def _create_interface(self):

        with gr.Blocks() as demo:
            gr.Markdown(""" ## The Signal and Image Processing Group Computer Vision Toolbox ![](https://sipg.isr.tecnico.ulisboa.pt/wp-content/uploads/2018/03/cropped-SIGP_logo_blue-copy_1x1.png)
            
            ## This website runs a set of algorithms on data uploaded buy users. Main tasks:
            
            - Detection of objects (YOLO) 
            - Tracking objects in sequences(To Be Released Soon!) 
            
            Support by ![](https://www.licentivos.pt/wp-content/uploads/2024/02/PRR.png) and others !""")

#---------------- TAB  imagem simples -----------------
            with gr.Tab("Yolo Single Image"):
                with gr.Row():
                    with gr.Column():
                        self.image_input = gr.Image(label="Input Image",type="numpy",interactive=True)
                        self.label_input = gr.Textbox(label="Input,Labels")
                        refresh_btn = gr.Button("🔄 Detect Objects", elem_id="refresh-btn")
                    with gr.Column():
                        self.image_output = gr.Image(label="Received Image", interactive=False)
                        self.label_output = gr.Textbox(label="Label", interactive=False)
                #   Metodos- Button click
                refresh_btn.click(
                    fn=self._update_acquire,
                    inputs=[self.image_input,self.label_input],
                    outputs=[self.image_output, self.label_output]
                )
#---------------- TAB Gallery para sequencias -----------------

            with gr.Tab("Yolo Image Sequence"):
                with gr.Row():
                    with gr.Column():
                        self.image_input_gallery = gr.Gallery(label="Input Images",type="numpy",interactive=True)
                        self.label_input_gallery = gr.Textbox(label="User Input")
                        with gr.Row():
                            run_yolo_detseq_btn = gr.Button("🔄 Detect Objects")
                            run_yolo_trackseq_btn = gr.Button("🔄 Track Objects")
                    with gr.Column():
                        self.image_output_gallery = gr.Gallery(label="Detected Objects",type="numpy")
                        self.label_output_gallery = gr.Textbox(label="Messages and Data", interactive=False)
                #   Metodos- Button click
                run_yolo_detseq_btn.click(
                    fn=self._update_sequence,
                    inputs=[self.image_input_gallery,self.label_input_gallery],
                    outputs=[self.image_output_gallery, self.label_output_gallery]
                )            
        return demo
#----------  Update ---------------
    def _update_sequence(self,img,label):
        
        if img is None:
            return None, "No image"
        #if there is an input image, process and wait for the answer
        try:
            with self.lock:    
                with open(self.input_data_file, 'wb') as f:
                    pickle.dump({"gradio":[img,label],"command":"sequence"},f)    
        except Exception as e:
            logging.error(f"Error in update_sequence SAVEMAT: {e}")
            return None, f"Error in update_sequence SAVEMAT: {e}"
        print("SAVED IMAGE ON PICKLE FILE")    
#------wait for response of the pipeline (imgs and json) -----------     
        while True:
            try:
                if os.path.exists(self.output_data_file):# Need to lock while loading
                    print("-----Images do DISPLAY ------")
                    with self.lock:
                        with open(self.output_data_file, 'rb') as f:
                            ret_data=pickle.load(f)
                        os.remove(self.output_data_file)
                        print(type(ret_data))
                    return ret_data[0],json.loads(ret_data[1])
                else:
                    time.sleep(.1)
            except Exception as e:
                logging.error(f"UPDATE_ACQUIRE: Error during loadmat: {e}")
                time.sleep(10)
                return None,"Error in data"

        
    def _update_acquire(self,img,label):
    #if user input an image and clicked on button, store data to be sent by grpc
    # and wait for result of processing
        if img is None:
            return None, "No image"
        #if there is an input image, process and wait for the answer
        try:
            with self.lock:
                
                with open(self.input_data_file, 'wb') as f:
                    pickle.dump({"gradio":[[img],label],"command":"single"},f)    
#                savemat(self.input_data_file,{"img":img,"label":label})
        except Exception as e:
            logging.error(f"Error in update_acquire SAVEMAT: {e}")
            return None, f"Error in update_acquire SAVEMAT: {e}"
#------wait for response of the pipeline (imgs and json) -----------     
        while True:
            try:
                if os.path.exists(self.output_data_file):# Need to lock while loading
                    with self.lock:
                        with open(self.output_data_file, 'rb') as f:
                            ret_data=pickle.load(f)
                        os.remove(self.output_data_file)
                    return ret_data[0],json.dumps(json.loads(ret_data[1]),indent=4)
                else:
                    time.sleep(.1)
            except Exception as e:
                logging.error(f"UPDATE_ACQUIRE: Error during loadmat: {e}")
                time.sleep(1)
                return None,"Error in data"

    def update(self, image_bytes: bytes, label: str):
        try:
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            with self.lock:
                 with open(self.output_data_file, 'wb') as f:
                    pickle.dump([image_np,label],f)  
        except Exception as e:
            logging.error(f"Error in update: {e}")
            image_np = np.full((500, 500, 3),255, dtype = np.uint8)
            label="Error"
            with self.lock:
                savemat(self.output_data_file,{"img":image_np,"label":label})

            
#----------- 
    def launch(self,share=True, server_name="0.0.0.0", server_port=7860):
        self.interface.launch(share=share, server_name=server_name, server_port=server_port)

