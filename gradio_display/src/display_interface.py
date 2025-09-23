import gradio as gr
import numpy as np
import cv2
import os
import time
import logging
import json
import threading
import pickle
from utils import write_list_to_temp,force_resize_image, extract_frames_resize_video
from utils import getdictrowsfromjson, getrowsfromjson, getfromkey
import tempfile

_IMG_MAX_SIZE = 640
   
#------------------------  Gradio DISPLAY OBJECT ---------      

class GradioDisplay:
    def __init__(self, tmp_data_folder="/tmp", lockfile=None):
        self.tmp_data_folder = tmp_data_folder
        self.lock = lockfile
        self.output_files = {}

        # central registry of in/out files by algorithm
        self.file_map = {}
        for algo in ["yolo", "vggt"]:
            self.file_map[algo] = {
                "in": os.path.join(tmp_data_folder, f"{algo}_in.mat"),
                "out": os.path.join(tmp_data_folder, f"{algo}_out.mat"),
            }
        logging.info(f"Initialized file_map keys: {list(self.file_map.keys())}")


        # cleanup old files
        for algo, paths in self.file_map.items():
            for f in paths.values():
                if os.path.exists(f):
                    os.remove(f)

        tempfile.tempdir = tmp_data_folder
        self.interface = self._create_interface()

# ------------------------------------------------------------------
    def _get_files(self, algo: str):
        """Return (input_file, output_file) for a given algorithm."""
        return self.file_map[algo]["in"], self.file_map[algo]["out"]

    def _write_request(self, algo: str, data: dict):
        in_file, _ = self._get_files(algo)
        with self.lock:
            with open(in_file, "wb") as f:
                pickle.dump(data, f)

        return in_file

    def _wait_for_response(self, algo: str, transform_fn=None):
        """Wait for response file of algo, optional transform on loaded data."""
        _, out_file = self._get_files(algo)
        while True:
            if os.path.exists(out_file):
                with self.lock:
                    with open(out_file, "rb") as f:
                        ret_data = pickle.load(f)
                    os.remove(out_file)
                return transform_fn(ret_data) if transform_fn else ret_data
            else:
                time.sleep(0.5)

# ------------------------------------------------------------------

#-------Launch the server ---------
    
    def launch(self,share=True, server_name="0.0.0.0", server_port=7860):
        self.interface.launch(
            share=share, server_name=server_name, server_port=server_port
            )
        
#--------  Create Gradio Interface --------
    
    def _create_interface(self):

        with gr.Blocks(delete_cache=(300,600),title="SIPg Toolbox") as demo:
            gr.Markdown("""![Deu asneira](https://drive.sipg.tecnico.ulisboa.pt/s/nrGaqCjrTbQTPY4/preview)""")

#---------------- TAB  Main TAB-----------------
            with gr.Tab("About "):

                gr.Markdown("""### This website provides computational services (algorithms) for signal processing tasks, some of them  known as “Artificial Intelligence” . 
                It’s purpose is mostly for demonstration and benchmark. 

                Underneath this interface all systems are processing pipelines built and run following AI4Europe standards, the [European platform for AI-on-demand](http://ai4europe.eu).
                
               | Services to members|  Algorithms | 
               |--------------|------------------------|               
               | [Online drawing (excalibur)](http://draw.sipg.tecnico.ulisboa.pt) |  Detection of objects (YOLO) |
               | [PDF manipulation](http://pdf.sipg.tecnico.ulisboa.pt) | Tracking objects in image sequences/videos (YOLO)|
               | [Storage on a private cloud](http://drive.sipg.tecnico.ulisboa.pt) | 3D reconstruction |
               |[Immich Photo](http://gphotos.sipg.tecnico.ulisboa.pt) | [Large Scale Camera Calibration](https://github.com/sipg-isr/vican) |
               |   | TBDeployed soon : distributed optim, sparse clustering ...  |
               
               This website was developped with support from project Smart Retail, funded through the PRR program
               ![Image not loaded](https://drive.sipg.tecnico.ulisboa.pt/s/WdaAsmyYT8B3QWw/preview)
               
              Other relevant supporters that partially funded this work: [Fundação para a Ciência e Tecnologia](http://www.fct.pt), 
              and ![Thales](https://drive.sipg.tecnico.ulisboa.pt/s/ABA8XPG7MjFgiaZ/preview) (Portugal)  """)
#---------------- TAB Gallery para sequencias -----------------

            with gr.Tab("Detection/Tracking for Image Sequences"):
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
#---------------- TAB VGGT -----------------

            with gr.Tab("3D reconstruction with VGGT"):
                with gr.Row():
                    with gr.Column():
                        self.vggt_image_input_gallery = gr.Gallery(label="Input Images",type="numpy",interactive=True)
                        self.vggt_label_input_gallery = gr.Textbox(label="User Input")
                        run_vggt_btn = gr.Button("🔄 Reconstruct 3D")
                    with gr.Column():
                        self.vggt_threeD_viewer = gr.Model3D()
                        self.vggt_conf_threshold = gr.Slider(0, 50, value=20, step=1, label="Confidence Threshold")

            #callbacks
            # Hook up callbacks
            run_yolo_detseq_btn.click(
                fn=self._update_sequence,
                inputs=[self.image_input_gallery, self.label_input_gallery],
                outputs=[
                    self.image_output_gallery,
                    self.label_output_gallery,
                    self.output_files_gallery,
                ],
            )
            run_yolo_trackseq_btn.click(
                fn=self._update_trackingsequence,
                inputs=[self.image_input_gallery, self.label_input_gallery],
                outputs=[
                    self.image_output_gallery,
                    self.label_output_gallery,
                    self.output_files_gallery,
                ],
                concurrency_limit=2,
            )
            run_vggt_btn.click(
                fn=self._update_vggt,
                inputs=[
                    self.vggt_image_input_gallery,
                    self.vggt_label_input_gallery,
                    self.vggt_conf_threshold,
                ],
                outputs=[self.vggt_threeD_viewer],
            )

        demo.queue(max_size=1)
        return demo

    def _update_vggt(self, img, label, conf_threshold, request: gr.Request):
        if img is None:
            return None
        logging.info("3D reconstruction request received")

        self._write_request(
            "vggt",
            {
                "gradio": [img, label],
                "command": "3d_infer",
                "parameters": {"conf_threshold": conf_threshold},
            },
        )
        logging.info(f"written in:{self._get_files('vggt')}")

        def transform(ret_data):
            glb_file_path = os.path.join(
                self.tmp_data_folder, f"vggt_{request.session_hash}.glb"
            )
            with open(glb_file_path, "wb") as f:
                f.write(ret_data[0])
            return glb_file_path

        return self._wait_for_response("vggt", transform_fn=transform)
 
#----------  DETECT Update ---------------
    def _update_sequence(self, img, label, request: gr.Request):

        #start by clearing previous outputs for this session
        if request.session_hash in self.output_files:
            self.output_files[request.session_hash]["files"] = {}
        else:
            self.output_files.update({request.session_hash: {"files": {}}})

        if img is None:
            return None, "No images uploaded!", None
        try:
            if isinstance(img[0][0], str):  # video path
                video = img[0][0]
                galeria = [
                    [x, None] for x in extract_frames_resize_video(video, _IMG_MAX_SIZE)
                ]
            else:
                galeria = [
                    [force_resize_image(x, _IMG_MAX_SIZE), y] for (x, y) in img
                ]
        except Exception as e:
            tmp = f"Image format not accepted: {e}"
            logging.error(tmp)
            return None, tmp, None

        self._write_request(
            "yolo",
            {"gradio": [galeria, label], "command": "detectsequence", "parameters": " "},
        )
        logging.info(f"written in:{self._get_files('yolo')}")

        def transform(ret_data):
            annotations = json.loads(ret_data[1])
            results = getrowsfromjson(getfromkey([annotations], "YOLO"))
            base, filepath = write_list_to_temp(
                results, prefix="detect_objects__", suffix=".csv"
            )
            basej, filepathj = write_list_to_temp(
                ret_data[1], prefix="detect_objects__", suffix=".json"
            )
            newfiles = {base: filepath, basej: filepathj}
            if request.session_hash in self.output_files:
                self.output_files[request.session_hash]["files"].update(newfiles)
            else:
                self.output_files.update({request.session_hash: {"files": newfiles}})
            return (
                ret_data[0],
                annotations,
                list(self.output_files[request.session_hash]["files"].values()),
            )

        return self._wait_for_response("yolo", transform_fn=transform)


    #---- TRACKING PROCESS -- TODO: collapse with detection into one single function.
    # Change form to have tracking as a tick (track/no track) in detection

    def _update_trackingsequence(self, img, label, request: gr.Request):

        #start by clearing previous outputs for this session
        if request.session_hash in self.output_files:
            self.output_files[request.session_hash]["files"] = {}
        else:
            self.output_files.update({request.session_hash: {"files": {}}})
            
        if img is None:
            return None, "No images uploaded!", None
        galeria = [[force_resize_image(x, _IMG_MAX_SIZE), y] for (x, y) in img]

        self._write_request(
            "yolo",
            {"gradio": [galeria, label], "command": "tracksequence", "parameters": ""},
        )

        def transform(ret_data):
            annotations = json.loads(ret_data[1])
            yoloannotations = getfromkey([annotations], "YOLO")
            results = getrowsfromjson(yoloannotations)
            base, filepath = write_list_to_temp(
                results, prefix="object_track__", suffix=".csv"
            )
            basej, filepathj = write_list_to_temp(
                ret_data[1], prefix="object_track__", suffix=".json"
            )
            newfiles = {base: filepath, basej: filepathj}
            if request.session_hash in self.output_files:
                self.output_files[request.session_hash]["files"].update(newfiles)
            else:
                self.output_files.update({request.session_hash: {"files": newfiles}})
            return (
                ret_data[0],
                annotations,
                list(self.output_files[request.session_hash]["files"].values()),
            )

        return self._wait_for_response("yolo", transform_fn=transform)
