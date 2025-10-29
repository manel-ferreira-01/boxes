import grpc
from concurrent import futures
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import sys
sys.path.append("../protos")
import pipeline_pb2 as yolo_pb2
import pipeline_pb2_grpc as yolo_pb2_grpc
from aux import wrap_value, unwrap_value

import cv2
import numpy as np
import json
import base64
from ultralytics import YOLO
import traceback

# VERIFY THE PORT NUMBER 
_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class PipelineService(yolo_pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self.model = YOLO("yolo11n.pt")  # Ensure this is YOLOv11
        logging.info("YOLOv11 model loaded.")
#       https://docs.ultralytics.com/modes/predict/#inference-arguments
 
       
    def __init__(self):
        self.model = YOLO("yolo11n.pt")
        logging.info("YOLOv11 model loaded.")

    def DetectSequence(self, request, context):
        """Run YOLO detection on a batch of images."""
        try:
            # ---- Parse and validate config ----
            if not request.config_json:
                raise ValueError("Missing config_json in request")

            try:
                config = json.loads(request.config_json)
            except json.JSONDecodeError:
                raise ValueError("config_json is not valid JSON")

            # ---- Extract and validate images ----
            img_list = unwrap_value(request.data.get("images", []))
            if not img_list:
                raise ValueError("No images provided in request.data['images']")

            # ---- Run detection ----
            annotated_images, detections = DetectSequence(self.model, img_list, config)

            # ---- Build successful response ----
            return yolo_pb2.Envelope(
                data={"images": wrap_value(annotated_images)},
                config_json=json.dumps({"YOLO": detections})
            )

        except Exception as e:
            logging.error(f"[DetectSequence] {e}")
            error_json = json.dumps({"YOLO_error": str(e)})
            return yolo_pb2.Envelope(
                config_json=error_json
            )

    def TrackSequence(self, request, context):
        """Run YOLO tracking on a sequence of images."""
        try:
            # ---- Parse and validate config ----
            if not request.config_json:
                return yolo_pb2.Envelope()

            try:
                config = json.loads(request.config_json)
            except json.JSONDecodeError:
                logging.error("config_json is not valid JSON")
            
            stream_countdown = config.get("stream", None)

            # ---- Extract and validate images ----
            try:
                img_list = unwrap_value(request.data.get("images", []))
            except Exception as e:
                # there is no image but there is a stream countdown == 0
                logging.info(f"YOLO: TrackSequence called with stream_countdown={stream_countdown}")
                if stream_countdown==0:
                    self.reset_tracker()
                return yolo_pb2.Envelope(config_json=request.config_json)
                
            # ---- Run tracking ----
            annotated_images, detections = TrackSequence(self.model, img_list, config)

            if stream_countdown == 0:
                self.reset_tracker()

            # ---- Build successful response ----
            return yolo_pb2.Envelope(
                data={"images": wrap_value(annotated_images)},
                config_json=json.dumps({"YOLO": detections})
            )

        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"[TrackSequence] Unhandled exception:\n{tb}")
            return yolo_pb2.Envelope()

    def reset_tracker(self):
        """Reset the YOLO tracker state."""
        try:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers[0].reset()
                logging.info("YOLO tracker reset successfully.")
        except Exception as e:
            logging.error(f"Failed to reset YOLO tracker: {e}")

               

#------------  Track each image --------        
    def Track(self, request, context):
            # Method just for one image TBD
        print("Not yet done ")
        
#--------------- Track a sequence of images --------
def DetectSequence(model, images, yolo_config ):
    """
    Process multiple images from request.images (repeated bytes),
    run YOLO inference, return annotated images and detections.
    """
    logging.info("YOLO: DetectSequence")
    annotated_images = []
    all_detections = []  # list of lists (per-image detections)
    # Check if there is any command and if it empty
    if "detectyoloconfig" in yolo_config:
        print(f"Yolo Configuration parameters {yolo_config['detectyoloconfig']}")
        #extract yoloconfig parameters here
        # Process each image
    for img_bytes in images:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., (2, 1, 0)]  # BGR→RGB            
        # Run YOLO inference
        results = model(img)
        # Annotate result

        #annotated_frame = cv2.cvtColor(
        #    results[0].plot(img=np.ascontiguousarray(results[0].orig_img)),
        #    cv2.COLOR_RGB2BGR
        #)

        annotated_frame = results[0].plot(img=np.ascontiguousarray(results[0].orig_img))
        
        _, labeled_image_bytes = cv2.imencode('.jpg', annotated_frame)
        annotated_images.append(labeled_image_bytes.tobytes())
        names = model.names
        # Collect detections
        detections_for_img = []
        for r in results:
            for b in r.boxes:
                detections_for_img.append({
                    "bbox": b.xyxy[0].tolist(),
                    "confidence": float(b.conf),
                    "class_id": int(b.cls),
                    "class_name": names.get(int(b.cls), f"class_{int(b.cls)}")#Retrieve name from class id
                })
        if not detections_for_img:
            detections_for_img = [{"no objects": ""}]
        all_detections.append(detections_for_img)
        
    return annotated_images, all_detections
        
    #----------- Track Sequence (input function) ---
        
def TrackSequence(model, images, yolo_config):
    """
    Process multiple images from images (repeated bytes),
    run YOLO inference, return annotated images and detections.

    must receive json 
    general: label=[{'aispgradio'},{other stuff}]
             l in label
                l['aispgradio']['empty'] - do nothing
    configtrack=l['aispgradio']['yolotrack'] 
        with commands
    configtrack['reset'] to start new features or continue previous sequence
    configtrack['kwargs'][*] comandos para track
    
    """
    annotated_images = []
    all_detections = []  # list of lists (per-image detections)
   
    stream_countdown = yolo_config.get("stream", None)

    
    for img_bytes in images:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., (2, 1, 0)]  # BGR→RGB

        results = model.track(source=img, persist=True)

        # Annotate result
        annotated_frame = cv2.cvtColor(
            results[0].plot(img=np.ascontiguousarray(results[0].orig_img)),
            cv2.COLOR_RGB2BGR
        )
        _, labeled_image_bytes = cv2.imencode('.jpg', annotated_frame)
        annotated_images.append(labeled_image_bytes.tobytes())

        # Collect detections
        detections_for_img = []
        names = model.names

        for r in results:
            for b in r.boxes:
                detections_for_img.append({
                    "track_id": int(b.id.item()) if b.id is not None else -1,
                    "bbox": b.xyxy[0].tolist(),
                    "confidence": float(b.conf),
                    "class_id": int(b.cls),
                    "class_name": names.get(int(b.cls), f"class_{int(b.cls)}")
                })


        all_detections.append(detections_for_img)

    return annotated_images, all_detections

def run_server(server):
    try:
        server_port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if server_port <= 0:
            logging.error('Port should be greater than 0')
            return None
    except ValueError:
        logging.exception('Unable to parse port')
        return None

    target = f'[::]:{server_port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'''Server started at {target}''')
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
    
    yolo_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    # Add reflection
    service_names = (
        yolo_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
