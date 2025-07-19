import grpc
import yolo_pb2
import yolo_pb2_grpc
import cv2

# Read image and encode as bytes
img = cv2.imread("test.jpg")
_, img_encoded = cv2.imencode('.jpg', img)
img_bytes = img_encoded.tobytes()

channel = grpc.insecure_channel('localhost:50051')
stub = yolo_pb2_grpc.YOLOserviceStub(channel)

response = stub.Detect(yolo_pb2.YOLORequest(image=img_bytes))
print("YOLO Results:", response.results)