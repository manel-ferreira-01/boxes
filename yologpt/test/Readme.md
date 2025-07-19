## 🚀 Python Client Example

The code in the yolotest.py script interacts with the Yolo service:
    - send one image
    - receive the json string and the annoted image

```python
import grpc
import yolo_pb2
import yolo_pb2_grpc
import cv2
import json
import numpy as np

# Load and encode image
img = cv2.imread("test.jpg")
_, img_encoded = cv2.imencode('.jpg', img)
img_bytes = img_encoded.tobytes()

# Connect to gRPC server on port 8061
#change server IP and port as appropriate

channel = grpc.insecure_channel('localhost:8061')
stub = yolo_pb2_grpc.YOLOserviceStub(channel)

# Send request
response = stub.Detect(yolo_pb2.YOLORequest(
    image=img_bytes,
    confidence_threshold=0.5
))

# Save labeled image
labeled_img = cv2.imdecode(np.frombuffer(response.labeled_image, np.uint8), cv2.IMREAD_COLOR)
cv2.imwrite("labeled_output.jpg", labeled_img)

# Parse detections
detections = json.loads(response.detections_json)
for d in detections:
    print(f"Detected {d['class_name']} with confidence {d['confidence']:.2f} at {d['bbox']}")
```
