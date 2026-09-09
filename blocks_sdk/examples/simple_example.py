#!/usr/bin/env python3
"""Simple example showing package-agnostic blocks_sdk calls."""
from blocks_sdk import Client, YOLO, TAPNext

# 1. Generic client (Send Envelope, Receive Envelope)
client = Client("localhost:8062")

print(f"Connected to box at {client.address}")

# Pass image file paths, PIL Images, or NumPy arrays directly!
# response = client.DetectSequence(images=["sample.jpg"], threshold=0.5)
# print("Response keys:", list(response.keys()))

# 2. Service wrappers
yolo = YOLO("localhost:8062")
tapnext = TAPNext("localhost:8061")

print("YOLO client initialized:", yolo.address)
print("TAPNext client initialized:", tapnext.address)
