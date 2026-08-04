#!/usr/bin/env python3
"""Test script for TAPNext gRPC service with video output."""

import sys
sys.path.append("../protos")

import grpc
import pipeline_pb2, pipeline_pb2_grpc, aux
import json
import cv2
import numpy as np

def load_video_frames(video_path):
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def bytes_to_tensor(b: bytes):
    """Deserialize tensor from bytes."""
    import torch
    import io as bio
    return torch.load(bio.BytesIO(b), weights_only=False)

def main():
    import matplotlib.pyplot as plt
    
    # Connect to the service
    channel = grpc.insecure_channel('localhost:8061')
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
    
    print("Testing TAPNext point tracking service...")
    
    # Load test video frames
    video_path = "./apple.mp4"
    try:
        frames = load_video_frames(video_path)
        print(f"Loaded {len(frames)} frames from test video")
        orig_h, orig_w = frames[0].shape[:2]
    except Exception as e:
        print(f"Failed to load video: {e}")
        return
    
    # Build request with all frames
    frame_bytes_list = []
    for frame in frames:
        _, buf = cv2.imencode('.jpg', frame)
        frame_bytes_list.append(buf.tobytes())
    
    config = {
        "tapnext": {
            "command": "track",
            "parameters": {
                "grid_size": 30,
                "reset": False
            }
        },
        "stream": max(0, len(frames) - 1)
    }
    
    request = pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={"images": aux.wrap_value(frame_bytes_list)}
    )
    
    print(f"\nSending tracking request with {len(frames)} frames...")
    response = stub.Process(request)
    
    result = json.loads(response.config_json)
    print(f"Response status: {result}")
    
    if not response.data or "tracks" not in response.data:
        print("No tracks data in response")
        return
    
    # Deserialize tracks and visibles
    tracks_tensor = bytes_to_tensor(aux.unwrap_value(response.data["tracks"]))
    visibles_tensor = bytes_to_tensor(aux.unwrap_value(response.data["visibles"]))
    
    print(f"Tracks shape: {tracks_tensor.shape}")
    print(f"Visibles shape: {visibles_tensor.shape}")
    
    # Convert to numpy
    tracks_np = tracks_tensor.numpy()  # Shape: (num_frames, num_points, 2)
    print(tracks_np)
    visibles_np = visibles_tensor.numpy()
    
    # Output video path
    output_video_path = "./output_tracking.mp4"
    
    # Setup video writer
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_w, orig_h))
    
    print(f"\nDrawing tracks on {len(frames)} frames...")
    
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx].copy()
        frame_tracks = tracks_np[frame_idx]  # [num_points, 2]
        frame_visibles = visibles_np[frame_idx]  # [num_points]
        
        # Draw each track point
        for pt_idx in range(len(frame_tracks)):
            #print("coods", frame_visibles.shape)
            if not frame_visibles[pt_idx]:
                continue
                
            coords = frame_tracks[pt_idx]
            y, x = float(coords[0]), float(coords[1])
            
            # Skip invalid coordinates (NaN or None)
            if np.isnan(x) or np.isnan(y):
                continue
            
            # Scale coordinates to original frame size
            #scale_x, scale_y = orig_w / 256.0, orig_h / 256.0
            #x_scaled = int(x / scale_x)
            #y_scaled = int(y / scale_y)
            #print(x_scaled, y_scaled)
            
            # Color based on track ID
            color = tuple(int(c) for c in np.array(plt.cm.rainbow(pt_idx / len(frame_tracks)))[:3] * 255)
            
            cv2.circle(frame, (int(x), int(y)), radius=4, color=color, thickness=-1)
        
        out.write(frame)
    
    out.release()
    print(f"\n✓ Output video saved to: {output_video_path}")
    print("✓ All tests completed!")

if __name__ == "__main__":
    main()
