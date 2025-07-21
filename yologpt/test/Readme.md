## 🚀 Python Client Example

The notebooks show how to interact with the service

The code in the yolotest.py script interacts with the Yolo service:
    - send one image
    - receive the json string and the annoted image and stores in files

```python
import grpc
import yolo_pb2
import yolo_pb2_grpc
import cv2
from scipy.io import savemat

# Read image and encode as bytes
img = cv2.imread("/home/jovyan/jpc/code/boxes/Images/eiffel.png")
_,img_encoded = cv2.imencode('.jpg', img)
img_bytes = img_encoded.tobytes()

channel = grpc.insecure_channel('Mac.lan:8061')
stub = yolo_pb2_grpc.YOLOserviceStub(channel)

response = stub.Detect(yolo_pb2.YOLORequest(image=img_bytes,confidence_threshold=0.2))


with open("output.jpg","wb") as f:
    f.write(response.labeled_image)
detections=eval(response.detections_json)
savemat("detections.mat",{"detections":detections})

channel.close()
```
