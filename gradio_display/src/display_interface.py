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
import subprocess

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
                time.sleep(0.01)

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
                        self.image_output_gallery = gr.Gallery(label="Detected Objects", type="numpy", visible=False)
                        self.track_output_video = gr.Video(label="Tracked Video Preview", visible=False)
                        self.label_output_gallery = gr.Textbox(label="Messages and Data", interactive=False)
                        self.output_files_gallery = gr.Files(file_count="multiple", label="Files: Processed Data")

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
                        self.vggt_mask_sky = gr.Checkbox(label="Mask Sky", value=False, visible=False)
                        self.vggt_out_matfile = gr.File(label="Output .mat file")

            #callbacks
            # Hook up callbacks
            run_yolo_detseq_btn.click(
                fn=self._update_sequence,
                inputs=[self.image_input_gallery, self.label_input_gallery],
                outputs=[
                    self.image_output_gallery,   # gallery
                    self.track_output_video,     # video
                    self.label_output_gallery,
                    self.output_files_gallery,
                ],
            )
            run_yolo_trackseq_btn.click(
                fn=self._update_trackingsequence,
                inputs=[self.image_input_gallery, self.label_input_gallery],
                outputs=[
                    self.image_output_gallery,   # gallery
                    self.track_output_video,     # video
                    self.label_output_gallery,
                    self.output_files_gallery,
                ],
            )

            run_vggt_btn.click(
                fn=self._update_vggt,
                inputs=[
                    self.vggt_image_input_gallery,
                    self.vggt_label_input_gallery,
                    self.vggt_conf_threshold,
                    self.vggt_mask_sky,
                ],
                outputs=[self.vggt_threeD_viewer,
                         self.vggt_out_matfile],
            )

        demo.queue(max_size=1)
        return demo

    def _update_vggt(self, img, label, conf_threshold, mask_sky: bool, request: gr.Request):
        if img is None:
            return None
        logging.info("3D reconstruction request received")

        self._write_request(
            "vggt",
            {
                "gradio": [img, label],
                "command": "3d_infer",
                "parameters": {"conf_threshold": conf_threshold, 
                               "device":"cpu",
                               "mask_sky": mask_sky},
            },
        )
        logging.info(f"written in:{self._get_files('vggt')}")

        def transform(ret_data):
            glb_file_path = os.path.join(
                self.tmp_data_folder, f"vggt_{request.session_hash}.glb"
            )
            with open(glb_file_path, "wb") as f:
                f.write(ret_data[0])

            mat_file_path = os.path.join(
                self.tmp_data_folder, f"vggt_{request.session_hash}.mat"
            )
            with open(mat_file_path, "wb") as f:
                f.write(ret_data[1].read())  # .mat file bytes

            return glb_file_path, mat_file_path

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
                gr.update(value=ret_data[0], visible=True),   # show Gallery
                gr.update(visible=False),                     # hide Video
                json.dumps(annotations),                      # safe string for Textbox
                list(self.output_files[request.session_hash]["files"].values()),
            )

        return self._wait_for_response("yolo", transform_fn=transform)


    #---- TRACKING PROCESS -- TODO: collapse with detection into one single function.
    # Change form to have tracking as a tick (track/no track) in detection

    def _update_trackingsequence(self, img, label, request: gr.Request):
        # start by clearing previous outputs for this session
        if request.session_hash in self.output_files:
            self.output_files[request.session_hash]["files"] = {}
        else:
            self.output_files[request.session_hash] = {"files": {}}

        if img is None:
            return None, "No images uploaded!", None

        try:
            # --- Detect video or image sequence ---
            if isinstance(img[0][0], str):  # video path
                video = img[0][0]
                frames = extract_frames_resize_video(video, _IMG_MAX_SIZE)
                galeria = [[f, None] for f in frames]
            else:  # gallery of images
                galeria = [[force_resize_image(x, _IMG_MAX_SIZE), y] for (x, y) in img]
        except Exception as e:
            tmp = f"Input format not accepted: {e}"
            logging.error(tmp)
            return None, tmp, None

        total_frames = len(galeria)
        all_annotations = []
        all_files = {}
        annotated_frames = []
        all_detections = []        # rows for CSV


        # --- Loop over frames ---
        for idx, (frame, y) in enumerate(galeria, start=1):
            countdown = total_frames - idx  # goes to 0 at the last frame

            self._write_request(
                "yolo",
                {
                    "gradio": [[[frame, y]], label],
                    "command": "tracksequence",
                    "parameters": {"stream": countdown},
                },
            )

            def transform(ret_data, _frame_idx=idx):
                annotations = json.loads(ret_data[1])
                yoloannotations = getfromkey([annotations], "YOLO")

                # rows already have frame index as first element
                rows = list(getrowsfromjson(yoloannotations))
                all_detections.extend(rows)

                all_annotations.append(annotations)
                return ret_data[0][0]  # annotated numpy image

            annotated_frame = self._wait_for_response("yolo", transform_fn=transform)
            annotated_frames.append(annotated_frame)

        # --- Write merged CSV + JSON ---
        base, filepath_csv = write_list_to_temp(
            all_detections, prefix="object_track__", suffix=".csv"
        )
        basej, filepath_json = write_list_to_temp(
            json.dumps(all_annotations), prefix="object_track__", suffix=".json"
        )

        self.output_files[request.session_hash]["files"].update(
            {base: filepath_csv, basej: filepath_json}
    )

        # --- Write stitched video preview ---
        h, w = annotated_frames[0].shape[:2]
        out_path = os.path.join(
            self.tmp_data_folder, f"track_preview_{request.session_hash}.mp4"
        )
        # Step 1: Write to AVI
        tmp_avi = out_path.replace(".mp4", ".avi")
        writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (w, h))
        for f in annotated_frames:
            writer.write(f)
        writer.release()

        # Step 2: Convert to H.264 MP4
        cmd = [
            "ffmpeg", "-y", "-i", tmp_avi,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            out_path
        ]
        subprocess.run(cmd, check=True)


        # Save output files for session
        self.output_files[request.session_hash]["files"].update(all_files)

        return (
            gr.update(visible=False),                        # hide Gallery
            gr.update(value=out_path, visible=True),         # show Video
            json.dumps(all_annotations),
            list(self.output_files[request.session_hash]["files"].values()),
        )

