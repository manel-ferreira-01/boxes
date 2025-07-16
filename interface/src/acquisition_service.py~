import grpc
import acquisition_pb2
import acquisition_pb2_grpc
from PIL import Image
import io

def save_image(img_bytes, filename="received.png"):
    with open(filename, "wb") as f:
        f.write(img_bytes)

def run_client():
    channel = grpc.insecure_channel('localhost:50051')
    stub = acquisition_pb2_grpc.AcquisitionServiceStub(channel)
    response = stub.acquire(acquisition_pb2.AcquireRequest())
    print("Received label:", response.label)
    save_image(response.image)

if __name__ == '__main__':
    run_client()
    
