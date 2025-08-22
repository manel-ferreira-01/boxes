
import grpc
from concurrent import futures
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import yolo_pb2
import yolo_pb2_grpc
import cv2
import numpy as np
import json
import base64
from ultralytics import YOLO

# VERIFY THE PORT NUMBER 
_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class YOLOServiceServicer(yolo_pb2_grpc.YOLOserviceServicer):
    def __init__(self):
        self.model = YOLO("yolo11n.pt")  # Ensure this is YOLOv11
#       https://docs.ultralytics.com/modes/predict/#inference-arguments
 
    def Detect(self, request, context):
        # Decode image from bytes
        try:
            nparr = np.frombuffer(request.image, np.uint8)
            print(nparr.shape)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[...,(2,1,0)]
            # Run YOLO inference
#--------> Parse requests.yolo_config (will have config values and other stuff)

            results = self.model(img)
            # Draw boxes on image + convert to BGR to use opencv
            annotated_frame = cv2.cvtColor(results[0].plot(img=np.ascontiguousarray(results[0].orig_img)),cv2.COLOR_RGB2BGR)
            # Encode image with boxes to bytes
            _, labeled_image_bytes = cv2.imencode('.jpg', annotated_frame)
            # Extract detection results
            detections = []
            for r in results:
                boxes = r.boxes
                for b in boxes:
                    box = b.xyxy[0].tolist()
                    conf = float(b.conf)
                    cls = int(b.cls)
                    detections.append({
                        "bbox": box,
                        "confidence": conf,
                        "class_id": cls
                    })
            if detections is None:
                detections={"no objects":""}
            #detections.append({"image": base64.b64encode(nparr).decode('utf-8')})
            # Convert detections to JSON string
        except:
            logging.error('DETECT:Image Not valid')
            _,labeled_image_bytes=cv2.imencode('.jpg',np.zeros((2,2,3),dtype='uint8'))
            detections={"yoloerror":"DETECT: Image not valid"}

        if request.yolo_config_json:
            detections_json = json.dumps([detections,json.loads(request.yolo_config_json)])
        else:
            detections_json = json.dumps(detections)
        logging.info(f"{detections_json}")    
        return yolo_pb2.YOLOResponse(
            labeled_image=labeled_image_bytes.tobytes(),
            detections_json=detections_json
        )
    
    def DetectSequence(self, request, context):
        """
        Process multiple images from request.images (repeated bytes),
        run YOLO inference, return annotated images and detections.
        """
        annotated_images = []
        all_detections = []  # list of lists (per-image detections)
        try:
            # Check if there is any command and if it empty
            if request.yolo_config_json:
                label=json.loads(request.yolo_config_json)
                for l in label:
                    if type(l) is dict:
                        if "aispgradio" in l.keys():
                            if "empty" in l['aispgradio'].keys():
                                return yolo_pb2.YOLOGradioSeq(
                                    images=[bytes(1)],
                                    yolo_config_json=request.yolo_config_json
                                )
                            elif "yoloconfig" in l['aispgradio']:
                                print(f"Yolo Configuration parameters {l['aispgradio']['yoloconfig']}")
                                #extract yoloconfig parameters here
            else:
               logging.error(f'----label is empty {request.yolo_config_json}  ')
                
    
            # Process each image
            for img_bytes in request.images:
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., (2, 1, 0)]  # BGR→RGB
                    
                # Run YOLO inference
                results = self.model(img)
    
                # Annotate result
                annotated_frame = cv2.cvtColor(
                    results[0].plot(img=np.ascontiguousarray(results[0].orig_img)),
                    cv2.COLOR_RGB2BGR
                )
                _, labeled_image_bytes = cv2.imencode('.jpg', annotated_frame)
                annotated_images.append(labeled_image_bytes.tobytes())
    
                # Collect detections
                detections_for_img = []
                for r in results:
                    for b in r.boxes:
                        detections_for_img.append({
                            "bbox": b.xyxy[0].tolist(),
                            "confidence": float(b.conf),
                            "class_id": int(b.cls)
                        })
                if not detections_for_img:
                    detections_for_img = [{"no objects": ""}]
    
                all_detections.append(detections_for_img)
    
        except Exception as e:
            logging.error(f'DETECT: Image processing error: {e}')
            _, labeled_image_bytes = cv2.imencode('.jpg', np.zeros((2, 2, 3), dtype='uint8'))
            annotated_images = [labeled_image_bytes.tobytes()]
            all_detections = {"ErrorYolo": "DETECT: Image not valid"}
    
        # Build detections JSON
        if request.yolo_config_json:
#            print("LABEL ----")
#            print(label)
            label.append({"YOLO": all_detections})
            detections_json = json.dumps(label)
 #           print(detections_json)
 #           print("DETECTIONS-----")
        else:
            detections_json = json.dumps([{"YOLO": all_detections}])

 #       print("------------returning from YOLO ---- ")
        
        return yolo_pb2.YOLOGradioSeq(
            images=annotated_images,
            yolo_config_json=detections_json
        )
    
    
    def Track(self, request, context):
            # Decode image
            frame = cv2.imdecode(np.frombuffer(request.image, np.uint8), cv2.IMREAD_COLOR)
            
            # Parse track config JSON
            try:
                config = json.loads(request.track_config_json or "{}")
                if config:
                    configyolo=config[1]
                    if config[0].pop("reset",None):#process commands (TODO)
                        self.model.predictor.trackers[0].reset() #reset tracker ID's
                else :
                    configyolo={}

            except json.JSONDecodeError:
                config = {}
                logging.exception("JSON not valid ")
            # Run tracking
            results = self.model.track(source=frame, persist=True, **configyolo)

            names = self.model.names
            detections = []

            for r in results:
                for b in r.boxes:
                    track_id = int(b.id.item()) if b.id is not None else -1
                    class_id = int(b.cls)
                    class_name = names.get(class_id, f"class_{class_id}")
                    conf = float(b.conf)
                    box = b.xyxy[0].tolist()
                    detections.append({
                        "track_id": track_id,
                        "bbox": box,
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": class_name
                    })

            annotated = results[0].plot()
            _, labeled_bytes = cv2.imencode('.jpg', annotated)

            return yolo_pb2.YOLOResponse(
                labeled_image=labeled_bytes.tobytes(),
                detections_json=json.dumps(detections)
            )

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
        options= [('grpc.max_send_message_length', 1024 * 1024 * 1024), 
                  ('grpc.max_receive_message_length', 1024 * 1024 * 1024)])
    
    yolo_pb2_grpc.add_YOLOserviceServicer_to_server(YOLOServiceServicer(), server)

    # Add reflection
    service_names = (
        yolo_pb2.DESCRIPTOR.services_by_name['YOLOservice'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
