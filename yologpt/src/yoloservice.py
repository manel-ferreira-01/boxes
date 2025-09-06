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
 
       
    def AllProcessing(self, request, context):
        """
        Process multiple images from request.images (repeated bytes),
        run YOLO inference, return annotated images and detections.
        """
        # Check if there is any command and if it empty
        
        if request.yolo_config_json:
            label=json.loads(request.yolo_config_json)
            for l in label:
                if type(l) is dict:
                    if "aispgradio" in l.keys():
                        comm=l['aispgradio'].keys()
                        logging.info(f"YOLO AllProcessing: {comm}")
                        try:
                            #------- Empty Packet , Do nothing ---------
                            if "empty" in l['aispgradio'].keys():
                                return yolo_pb2.YOLOGradioSeq(
                                    images=[bytes(1)],
                                    yolo_config_json=request.yolo_config_json
                                )
                            #-------- Process Commands ----------
                            elif "command" in l['aispgradio']:
                                print(f"Yolo command:  {l['aispgradio']['command']}")
                                logging.info(f'The command is : {request.yolo_config_json}  ')
                                #----- Detect in a sequence -------
                                if "detectsequence" in l['aispgradio']['command']:
                                    annotated_images,all_detections=DetectSequence(self.model,request.images,l['aispgradio'])
                                   
                                #----- Track a sequence -------
                                elif "tracksequence" in l['aispgradio']['command']:
                                    if hasattr(self.model.predictor,"trackers"):
                                        self.model.predictor.trackers[0].reset() 
                                    annotated_images,all_detections=TrackSequence(self.model,request.images,l['aispgradio'])
                                    print("YOLO: Foi ao tracksequence")       
                                    
                                elif "single" in l['aispgradio']['command']:
                                    annotated_images,all_detections=DetectSequence(self.model,request.images,l['aispgradio'])
                                    print("YOLO: Foi ao Detect single")            

                                else :
                                    logging.error(f'----label message is empty or no command {request.yolo_config_json}  ')
                                    return yolo_pb2.YOLOGradioSeq(
                                        images=[bytes(1)],
                                        yolo_config_json=json.dumps({"Error":"Method AllProcessing requires json command"})
                                    )
                        except Exception as e:
                            logging.error(f'AllProcessing error:  {e}')
                            _, labeled_image_bytes = cv2.imencode('.jpg', np.zeros((5, 5, 3), dtype='uint8'))
                            annotated_images = [labeled_image_bytes.tobytes()]
                            all_detections = {"ErrorYolo": f"TRACKING: {e}"}
                            
                    label.append({"YOLO": all_detections})
                    detections_json=json.dumps(label)
                    return yolo_pb2.YOLOGradioSeq(
                        images=annotated_images,yolo_config_json=detections_json
                    )
        else:
            logging.error(f'----label message is empty or no command {request.yolo_config_json}  ')
            return yolo_pb2.YOLOGradioSeq(
                images=[bytes(1)],
                yolo_config_json=json.dumps({"YOLO":"error: Method AllProcessing requires json command"})
            )
               

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
        annotated_frame = cv2.cvtColor(
            results[0].plot(img=np.ascontiguousarray(results[0].orig_img)),
            cv2.COLOR_RGB2BGR
        )
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
    # Check if there is any command and if it empty
    if "trackyoloconfig" in yolo_config:
        print(f"Yolo Configuration parameters {yolo_config['trackyoloconfig']}")
        configtrack=yolo_config['trackyoloconfig']
        #extract yoloconfig parameters here
        # Process each image
    else:
        configtrack={} # set the default parameters

    #------- parse yolo commands --------
    if configtrack.pop("reset",None):#process commands (TODO)
# ---->      --MUST CHECK trackers  EXISTS 
        model.predictor.trackers[0].reset() #reset tracker ID's
    
    for img_bytes in images:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)[..., (2, 1, 0)]  # BGR→RGB

        results = model.track(source=img, persist=True, **configtrack)

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
                track_id = int(b.id.item()) if b.id is not None else -1
                class_id = int(b.cls)
                class_name = names.get(class_id, f"class_{class_id}")
                conf = float(b.conf)
                box = b.xyxy[0].tolist()
                detections_for_img.append({
                    "track_id": track_id,
                    "bbox": box,
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name
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
