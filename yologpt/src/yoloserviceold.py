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
from ultralytics import YOLO

# VERIFY THE PORT NUMBER 
_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class YOLOServiceServicer(yolo_pb2_grpc.YOLOserviceServicer):
    def __init__(self):
        self.model = YOLO("yolo11n.pt")  # Ensure this is YOLOv11
        self.model.predictor.trackers[0].reset()
#       https://docs.ultralytics.com/modes/predict/#inference-arguments
 
    def Detect(self, request, context):
        # Decode image from bytes
        nparr = np.frombuffer(request.image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLO inference
        results = self.model(img)

        # Draw boxes on image
        annotated_frame = results[0].plot()

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

        # Convert detections to JSON string
        detections_json = json.dumps(detections)

        return yolo_pb2.YOLOResponse(
            labeled_image=labeled_image_bytes.tobytes(),
            detections_json=detections_json
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
        options= [('grpc.max_send_message_length', 512 * 1024 * 1024), 
                  ('grpc.max_receive_message_length', 512 * 1024 * 1024)])
    
    yolo_pb2_grpc.add_YOLOserviceServicer_to_server(YOLOServiceServicer(), server)

    # Add reflection
    service_names = (
        yolo_pb2.DESCRIPTOR.services_by_name['YOLOservice'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
