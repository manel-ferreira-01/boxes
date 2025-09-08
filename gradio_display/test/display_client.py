import grpc
import display_pb2
import display_pb2_grpc
import argparse
import cv2
import os

def encode_image(image_path: str) -> bytes:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()

def send_to_display_server(image_bytes: bytes, label: str, host: str = "localhost", port: int = 8161):
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = display_pb2_grpc.DisplayServiceStub(channel)
    
    request = display_pb2.DisplayRequest(
        label=label,
        image=image_bytes
    )
    
    response = stub.display(request)
    print("✅ Image and label sent to display server.")

def main():
    parser = argparse.ArgumentParser(description="Send image and label to gRPC display server")
    parser.add_argument("--image", required=True, help="Path to image file (jpg/png)")
    parser.add_argument("--label", required=True, help="Label text to display")
    parser.add_argument("--host", default="localhost", help="gRPC server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8161, help="gRPC server port (default: 8161)")

    args = parser.parse_args()

    try:
        image_bytes = encode_image(args.image)
        send_to_display_server(image_bytes, args.label, args.host, args.port)
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
