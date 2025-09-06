import gradio as gr
import numpy as np
import cv2
import os
import time
import logging
import json
import threading
import pickle
from utils import write_list_to_temp,force_resize_image
from utils import getdictrowsfromjson, getrowsfromjson, getfromkey
import tempfile

_IMG_MAX_SIZE = 640
   
#------------------------  Gradio DISPLAY OBJECT ---------      

class GradioDisplay:
    def __init__(self,tmp_data_folder="/tmp",lockfile=None):
        # self.(gradio) image_input, label_input,image_outpu,label_output
        self.image = None
        self.label = "User label (ex: token)"
        self.output_files_gallery=None
        self.input_data_file=tmp_data_folder+"/in.mat"
        self.output_data_file=tmp_data_folder+"/out.mat"
        self.output_files={}
        self.tmp_data_folder = tmp_data_folder
        self.lock=lockfile
        
        self.interface = self._create_interface()
        if os.path.exists(self.input_data_file):
            os.remove(self.input_data_file)
        if os.path.exists(self.output_data_file):
            os.remove(self.output_data_file)
        tempfile.tempdir=tmp_data_folder
#-----------METHODS -----------------

#-------Launch the server ---------
    
    def launch(self,share=True, server_name="0.0.0.0", server_port=7860):
        self.interface.launch(share=share, server_name=server_name, server_port=server_port)
        
#--------  Create Gradio Interface --------
    
    def _create_interface(self):

        with gr.Blocks(delete_cache=(300,600),title="SIPg Toolbox") as demo:
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
                        self.output_files_gallery= gr.Files(file_count="multiple", label="Files: Processed Data")
#---Callbacks   Metodos- Button click
                run_yolo_detseq_btn.click(
                    fn=self._update_sequence,
                    inputs=[self.image_input_gallery,self.label_input_gallery],
                    outputs=[self.image_output_gallery, self.label_output_gallery,self.output_files_gallery]
                )
                run_yolo_trackseq_btn.click(
                    fn=self._update_trackingsequence,
                    inputs=[self.image_input_gallery,self.label_input_gallery],
                    outputs=[self.image_output_gallery, self.label_output_gallery,self.output_files_gallery],
                    concurrency_limit=2
                )
                
            self.output_files_gallery.download(self._delete_results_files,None,self.output_files_gallery)
        demo.queue(max_size=1)
        return demo

#---Callback  Delete temporary files ---------
    
    def _delete_results_files(self,download_data: gr.DownloadData,request:gr.Request):
        file_name = os.path.basename(download_data.file.path)
        if request.session_hash:
            toremove=self.output_files[request.session_hash]["files"].pop(file_name,None)
            if toremove:
                if os.path.exists(toremove):
                    os.remove(toremove)
            return list(self.output_files[request.session_hash]["files"].values())

#----------- COmmands for yolo container
#    print(f"Yolo command:  {l['aispgradio']['command']}")
#    if "detectsequence" in l['aispgradio']['command']:
#        return DetectSequence(self,request,context)
#    elif "tracksequence" in l['aispgradio']['command']:
 
#----------  DETECT Update ---------------
    def _update_sequence(self,img,label,request:gr.Request):
        
        if img is None:
            return None, "No images uploaded!",None
        else:
            galeria=[[force_resize_image(x,_IMG_MAX_SIZE),y] for (x,y) in img]

        #if there is an input image, process and wait for the answer
        try:
            with self.lock:    
                with open(self.input_data_file, 'wb') as f:
                    pickle.dump({"gradio":[galeria,label],"command":"detectsequence"},f)    
            
#------wait for response of the pipeline (imgs and json) -----------     
            while True:
                if os.path.exists(self.output_data_file):# Need to lock while loading
                    with self.lock:
                        with open(self.output_data_file, 'rb') as f:
                            ret_data=pickle.load(f)
                        os.remove(self.output_data_file)
# ---desdobra json para csv
                    annotations= json.loads(ret_data[1])                    
                    results=getrowsfromjson(getfromkey(annotations,"YOLO"))
                   
                    base,filepath= write_list_to_temp(results,prefix="detect_objects__",suffix=".csv")
                    basej,filepathj= write_list_to_temp(ret_data[1],prefix="detect_objects__",suffix=".json")
                    newfiles={base:filepath,basej:filepathj}
                    
                    if request.session_hash in  self.output_files:
                        self.output_files[request.session_hash]["files"].update(newfiles)
                    else:
                        self.output_files.update({request.session_hash:{"files":newfiles}})
                        
                    return ret_data[0],annotations,list(self.output_files[request.session_hash]["files"].values())                
                else:
                    time.sleep(.5)
        except Exception as e:
            logging.error(f"Error in update_trackingsequence: {e}")
            time.sleep(10)
            return None, f"Error in update_trackingsequence : {e}",None

    #---- TRACKING PROCESS -- TODO: collapse with detection into one single function.
    # Change form to have tracking as a tick (track/no track) in detection

    def _update_trackingsequence(self,img,label,request:gr.Request):
        
        if img is None:
            return None, "No images uploaded!",None
        else:
            galeria=[[force_resize_image(x,_IMG_MAX_SIZE),y] for (x,y) in img]

        #if there is an input image, process and wait for the answer
        try:
            with self.lock:    
                with open(self.input_data_file, 'wb') as f:
                    pickle.dump({"gradio":[galeria,label],"command":"tracksequence"},f)    
            
#------wait for response of the pipeline (imgs and json) -----------     
            while True:
                if os.path.exists(self.output_data_file):# Need to lock while loading
                    with self.lock:
                        with open(self.output_data_file, 'rb') as f:
                            ret_data=pickle.load(f)
                        os.remove(self.output_data_file)
# ---desdobra json para csv
                    annotations= json.loads(ret_data[1])                    
                    results=getrowsfromjson(getfromkey(annotations,"YOLO"))
                   
                    base,filepath= write_list_to_temp(results,prefix="object_track__",suffix=".csv")
                    basej,filepathj= write_list_to_temp(ret_data[1],prefix="object_track__",suffix=".json")
                    newfiles={base:filepath,basej:filepathj}
                    
                    if request.session_hash in  self.output_files:
                        self.output_files[request.session_hash]["files"].update(newfiles)
                    else:
                        self.output_files.update({request.session_hash:{"files":newfiles}})
                        
                    return ret_data[0],annotations,list(self.output_files[request.session_hash]["files"].values())                
                else:
                    time.sleep(.5)
        except Exception as e:
            logging.error(f"Error in update_trackingsequence: {e}")
            time.sleep(10)
            return None, f"Error in update_trackingsequence : {e}",None
        
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
