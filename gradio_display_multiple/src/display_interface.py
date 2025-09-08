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
import itertools 

#------------------- UTILS -------------------    

def flatten(lst):
    return list(itertools.chain.from_iterable(map(flatten, lst)) if isinstance(lst, list) else [lst])

def getfromkey(lista,key):
    for l in lista:
        if key in l:
            return l[key]
    return None

def getrowsfromjson(seqdetections):
    rows=[]
    for i,img in enumerate(seqdetections) :
        for obj in img:
            if obj:
                tmp=list(obj.values())
                tmp.insert(0,i+1)
                rows.append(flatten(tmp))
    return rows
    
#------------------------  Gradio DISPLAY OBJECT ---------      
class GradioDisplay:
    def __init__(self,tmp_data_folder="/dev/shm",lockfile=None):
        # self.(gradio) image_input, label_input,image_outpu,label_output
        self.image = None
        self.label = "User label (ex: token)"
        self.input_data_file=tmp_data_folder+"/in.mat"
        self.output_data_file=tmp_data_folder+"/out.mat"
        self.lock=lockfile
        
        self.interface = self._create_interface()
        if os.path.exists(self.input_data_file):
            os.remove(self.input_data_file)
        if os.path.exists(self.output_data_file):
            os.remove(self.output_data_file)

#-------Launch the server 
    
    def launch(self,share=True, server_name="0.0.0.0", server_port=7860):
        self.interface.launch(share=share, server_name=server_name, server_port=server_port)

        
            
#--------  Create Gradio Interface --------
    
    def _create_interface(self):

        with gr.Blocks(delete_cache=(30,120),title="SIPg Toolbox") as demo:
            gr.Markdown("""![Deu asneira](https://drive.sipg.tecnico.ulisboa.pt/s/zkiEPD7qzCgyWkK/preview)       

            ### This website is under active development it may change at any time. For now, the main tasks available:

            - Detection of objects (YOLO) 
            - Tracking objects in image sequences/videos
            - 3D reconstruction with VGGT
            
            Support by many institutions, including relevant [defunct institutions](http://www.fct.pt), and ![Image not loaded](https://drive.sipg.tecnico.ulisboa.pt/s/WdaAsmyYT8B3QWw/preview)""")

#---------------- TAB  Main TAB-----------------
            with gr.Tab("Main Menu"):
                gr.Markdown("""## Upload a file, capture from your webcam or paste from clipboard""")
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
                run_yolo_trackseq_btn.click(
                    fn=self._update_trackingsequence,
                    inputs=[self.image_input_gallery,self.label_input_gallery],
                    outputs=[self.image_output_gallery, self.label_output_gallery]
                )
        demo.queue(max_size=1)
        return demo
#----------- COmmands for yolo container
#    print(f"Yolo command:  {l['aispgradio']['command']}")
#    if "detectsequence" in l['aispgradio']['command']:
#        return DetectSequence(self,request,context)
#    elif "tracksequence" in l['aispgradio']['command']:
 
#----------  DETECT Update ---------------
    def _update_sequence(self,img,label):
        
        if img is None:
            return None, "ERROR Detection: No images"
        #if there is an input image, process and wait for the answer
        try:
            with self.lock:    
                with open(self.input_data_file, 'wb') as f:
                    pickle.dump({"gradio":[img,label],"command":"detectsequence"},f)   
                print("Detection : Pickle written")
        except Exception as e:
            logging.error(f"Error in update_sequence : {e}")
            return None, f"Error in update_sequence : {e}"
            
#------wait for response of the pipeline (imgs and json) -----------     
        while True:
            try:
                if os.path.exists(self.output_data_file):# Need to lock while loading
#                    print("-----Images do DISPLAY: TRACKING ------")
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

    #---- TRACKING PROCESS -- TODO: collapse with detection into one single function.
    # Change form to have tracking as a tick (track/no track) in detection

    def _update_trackingsequence(self,img,label):
        
        if img is None:
            return None, "ERROR TRACKING: No images"
        #if there is an input image, process and wait for the answer
        try:
            with self.lock:    
                with open(self.input_data_file, 'wb') as f:
                    pickle.dump({"gradio":[img,label],"command":"tracksequence"},f)    
            
#------wait for response of the pipeline (imgs and json) -----------     
            while True:
                if os.path.exists(self.output_data_file):# Need to lock while loading
                    with self.lock:
                        with open(self.output_data_file, 'rb') as f:
                            ret_data=pickle.load(f)
                        os.remove(self.output_data_file)
# ---desdobra json para csv
                    results=getrowsfromjson(getfromkey(json.loads(ret_data[1]),"YOLO"))                    
                    return ret_data[0],results                   
                else:
                    time.sleep(.1)
        except Exception as e:
            logging.error(f"Error in update_trackingsequence: {e}")
            time.sleep(10)
            return None, f"Error in update_trackingsequence : {e}"

        
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
        except Exception as e:
            logging.error(f"Error in update: {e}")
            image_np = np.full((500, 500, 3),0, dtype = np.uint8)
            label="Display Error" #------- put messages as keys in label

        with self.lock:
             with open(self.output_data_file, 'wb') as f:
                pickle.dump([image_np,label],f)  
            
#----------- 
