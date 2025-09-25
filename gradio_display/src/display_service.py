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

    # ------------------------------------------------------
    # Helpers
    def _get_files(self, algo: str):
        return self.gradio_display.file_map[algo]["in"], self.gradio_display.file_map[algo]["out"]

    def _wait_for_request(self, algo: str):
        """Wait until the algorithm's in_file appears, then return its data."""
        in_file, _ = self._get_files(algo)
        while True:
            if os.path.exists(in_file):
                with lock:
                    with open(in_file, "rb") as f:
                        data = pickle.load(f)
                    os.remove(in_file)
                return data
            time.sleep(0.01)

    def _write_response(self, algo: str, payload):
        """Write output to the algorithm's out_file."""
        _, out_file = self._get_files(algo)
        with lock:
            with open(out_file, "wb") as f:
                pickle.dump(payload, f)

    def acquire(self, request, context):
        """Handles the acquire() loop for any algorithm request.
        Returns 'empty' if no file is available (non-blocking)."""
        #logging.info("DISPLAY : acquire request")
        time.sleep(0.01)
        for algo in self.gradio_display.file_map.keys():
            in_file, _ = self._get_files(algo)
            #logging.info(f"{in_file}")
            
            if os.path.exists(in_file):
                logging.info(f"DISPLAY : Found data file {in_file}, processing")
                with lock:
                    with open(in_file, "rb") as f:
                        gradio_data = pickle.load(f)
                    os.remove(in_file)

                # --- Encode images depending on command
                if gradio_data["command"] in ("detectsequence", "tracksequence"):
                    gg = gradio_data["gradio"][0]
                    image_bytes = [
                        cv2.imencode(".jpg", im)[1].tobytes()
                        for im in [pair[0] for pair in gg]
                    ]

                elif gradio_data["command"] == "3d_infer":
                    gg = gradio_data["gradio"][0]
                    image_bytes = [
                        cv2.imencode(".jpg", im)[1].tobytes()
                        for im in [pair[0] for pair in gg]
                    ]

                else:
                    logging.error("Unknown command in acquire")
                    image_bytes = []

                # --- Add metadata
                self.input_count += 1
                annotations = {
                    "command": gradio_data["command"],
                    "user": gradio_data["gradio"][1],
                    "input_count": self.input_count,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "parameters": gradio_data.get("parameters", {}),
                }

                return display_pb2.Envelope(
                    config_json=json.dumps({"aispgradio": annotations}),
                    data={"images": wrap_value(image_bytes)},
                )

            else:
                continue

        out_json = json.dumps({"aispgradio":  {"empty": "empty"}})
        _, tmp = cv2.imencode(".jpg", np.zeros((2, 2, 3), dtype="uint8"))
        image_bytes = [tmp.tobytes()]
        return display_pb2.Envelope(
            config_json=out_json, data={"images": wrap_value(image_bytes)}
        )


    def display_yolo(self, request, context):
        """Handles YOLO display (detect/track)."""
        #logging.info("DISPLAY : YOLO request")
        config_json = json.loads(request.config_json)
        time.sleep(0.01)  # slight delay to ensure file is written

        if "aispgradio" in config_json.keys():
            if "empty" in config_json["aispgradio"].keys():
                #logging.error("No aispgradio entry in config_json, returning empty response")
                return display_pb2.Envelope()
            elif config_json["aispgradio"]["command"] in ("detectsequence", "tracksequence"):
                images = [
                    cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                    for img in unwrap_value(request.data["images"])
                ]
                self._write_response("yolo", [images, request.config_json])
            else:
                #logging.error(f"Unknown command {config_json['aispgradio']['command']}, returning empty response")
                return display_pb2.Envelope()
        return display_pb2.Envelope()

    def display_vggt(self, request, context):
        """Handles VGGT display (3D reconstruction)."""
        #logging.info("DISPLAY : VGGT request")
        config_json = json.loads(request.config_json)
        time.sleep(0.01)  # slight delay to ensure file is written

        if "aispgradio" in config_json.keys():
            if "empty" in config_json["aispgradio"].keys():
                #logging.error("No aispgradio entry in config_json, returning empty response")
                return display_pb2.Envelope()
            elif config_json["aispgradio"]["command"] in ("3d_infer", ):
                glb_file = unwrap_value(request.data["glb_file"]) or None
                self._write_response("vggt", [glb_file, request.config_json])
            else:
                #logging.error(f"Unknown command {config_json['aispgradio']['command']}, returning empty response")
                return display_pb2.Envelope()
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



