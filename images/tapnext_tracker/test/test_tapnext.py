#!/usr/bin/env python3
"""Test script for TAPNext gRPC service."""

import sys
sys.path.append("/workspace/protos")

import grpc
from protos import pipeline_pb2, pipeline_pb2_grpc, aux
import json

def load_frame_bytes(video_path):
    """Load frames from a video file as bytes."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buf = cv2.imencode('.jpg', frame)
        frames.append(buf.tobytes())
    cap.release()
    return frames

def main():
    # Connect to the service
    channel = grpc.insecure_channel('localhost:8061')
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
    
    print("Testing TAPNext point tracking service...")
    
    # Load test video frames
    video_path = "./test/apple.mp4"
    try:
        frames = load_frame_bytes(video_path)
        print(f"Loaded {len(frames)} frames from test video")
    except Exception as e:
        print(f"Failed to load video: {e}")
        print("Testing with minimal request...")
    
    # Test 1: Track request with single frame
    if 'frames' in locals() and len(frames) > 0:
        config = {
            "tapnext": {
                "command": "track",
                "parameters": {
                    "grid_size": 32,
                    "reset": False
                }
            },
            "stream": max(0, len(frames) - 1)
        }
        
        request = pipeline_pb2.Envelope(
            config_json=json.dumps(config),
            data={"images": aux.wrap_value(frames)}
        )
        
        print(f"\nSending tracking request with {len(frames)} frames...")
        response = stub.Process(request)
        
        result = json.loads(response.config_json)
        print(f"Response status: {result}")
        
        if response.data:
            print("Response data keys:", list(response.data.keys()))
    
    # Test 2: Reset request
    print("\nTesting reset command...")
    config_reset = {
        "tapnext": {
            "command": "reset",
            "parameters": {"grid_size": 16}
        }
    }
    response_reset = stub.Process(pipeline_pb2.Envelope(config_json=json.dumps(config_reset)))
    result_reset = json.loads(response_reset.config_json)
    print(f"Reset response: {result_reset}")
    
    # Test 3: Empty request
    print("\nTesting empty request...")
    response_empty = stub.Process(pipeline_pb2.Envelope())
    print(f"Empty response config: {response_empty.config_json}")
    
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    main()
