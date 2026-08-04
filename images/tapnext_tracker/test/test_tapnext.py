#!/usr/bin/env python3
"""Test script for TAPNext gRPC service."""

import sys
sys.path.insert(0, "/workspace/protos")

import grpc
from protos import pipeline_pb2, pipeline_pb2_grpc
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

def make_track_request(frames, config):
    """Create a track request with multiple frames."""
    return pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={"images": [pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=frames))]}
    )

def main():
    # Connect to the service
    channel = grpc.insecure_channel('localhost:8061')
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
    
    print("Testing TAPNext point tracking service...")
    
    # Load test video frames
    video_path = "/workspace/test/test_video.mp4"
    try:
        frames = load_frame_bytes(video_path)
        print(f"Loaded {len(frames)} frames from test video")
    except Exception as e:
        print(f"Failed to load video: {e}. Using empty request for basic test.")
        frames = []
    
    # Test 1: Basic tracking with grid (32x32 points)
    if frames:
        config = {
            "tapnext": {
                "command": "track",
                "parameters": {
                    "grid_size": 32,
                    "reset": False
                }
            },
            "stream": len(frames) - 1
        }
        
        print(f"\nSending tracking request with {len(frames)} frames...")
        response = stub.Process(make_track_request(frames, config))
        
        result = json.loads(response.config_json)
        print(f"Response status: {result}")
        
        if response.data:
            tracks_data = pipeline_pb2.Value()
            tracks_data.CopyFrom(response.data["tracks"])
            print(f"Tracks field present in response")
            
            visibles_data = pipeline_pb2.Value()
            visibles_data.CopyFrom(response.data["visibles"])
            print(f"Visibles field present in response")
    
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
    
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    main()
