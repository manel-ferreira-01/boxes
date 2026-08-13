#!/usr/bin/env python3
"""Example of calling services - TAPNext with actual video."""
import sys
sys.path.insert(0, '/home/manuelf/boxes/blocks_sdk')

from blocks_sdk import TAPNext

# Initialize client
tapnext = TAPNext("localhost:8061")

print("=== TAPNext Video Tracking Example ===")
print(f"Service: {tapnext.address}")

apple_video = "/home/manuelf/boxes/images/tapnext_tracker/test/apple.mp4"

import os
if os.path.exists(apple_video):
    with open(apple_video, 'rb') as f:
        video_bytes = f.read()
    
    print(f"\nLoaded apple video: {len(video_bytes)} bytes")
    
    # TAPNext uses Process method
    response = tapnext.call(
        method="Process",
        data={"images": [video_bytes]},
        config_json='{"tapnext": {"parameters": {"grid_size": 32}}}'
    )
    
    from blocks_sdk import helpers
    
    print(f"\nResponse status: {response.config_json}")
    
    if response.data:
        for k, v in response.data.items():
            val = helpers.unwrap_value(v)
            print(f"Data '{k}': type={type(val).__name__}, size={len(val) if isinstance(val, list) else 'N/A'}")
else:
    print(f"Video not found: {apple_video}")

print("\n=== Example complete! ===")

# Show how to call other services
print("""
For YOLO service (port 8062), use:

from blocks_sdk import YOLO
yolo = YOLO("localhost:8062")

try:
    response = yolo.call(
        method="DetectSequence",
        data={"images": [img_bytes]},
        config_json='{"threshold": 0.5}'
    )
except ValueError as e:
    # Service may not have this method - check docs
    print(f"Note: {e}")
""")
