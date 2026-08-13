#!/usr/bin/env python3
"""Simple examples of calling services."""
import sys
sys.path.insert(0, '/home/manuelf/boxes/blocks_sdk')

from blocks_sdk import TAPNext, YOLO

# Initialize clients - use localhost:port mappings from docker-compose.yml
tapnext = TAPNext("localhost:8061")
yolo = YOLO("localhost:8062")

print(f"TAPNext client: {tapnext.address}")
print(f"YOLO client: {yolo.address}")

# Example usage (replace with actual image bytes):
# 
# # Load an image
# with open("image.jpg", "rb") as f:
#     img_bytes = f.read()
#
# # Call TAPNext to track points
# response = tapnext.track([img_bytes])
#
# # Call YOLO for detection
# results = yolo.detect([img_bytes], threshold=0.5)
