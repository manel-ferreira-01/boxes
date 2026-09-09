# blocks_sdk

A simple, package-agnostic Python SDK to invoke AI microservice container boxes over gRPC using standard Protobuf Envelopes (`send an envelope, receive an envelope`).

## Features

- **Package-Agnostic Image Handling**: Pass image file paths (`"photo.jpg"` or `Path("...")`), PIL Images, OpenCV/NumPy arrays, base64 strings, or raw `bytes` directly into the SDK without manual byte conversion code.
- **Universal Envelope Contract**: Built on top of the system-wide `pipeline.Envelope` protobuf format (`config_json` + `data` map of `Value`s). Send an Envelope, receive an unwrapped Python dictionary directly.
- **Zero Heavy Dependencies**: Requires only standard `grpcio`. No heavy required image libraries or dynamic server reflection overhead.
- **Dynamic Method Invocation**: Call any method exposed by a box (`client.DetectSequence(...)`, `client.Process(...)`, `client.similarity_check(...)`, `client.Forward(...)`).

## Quick Start

```python
from blocks_sdk import Client

# 1. Connect to any AI service box
client = Client("localhost:8062")

# 2. Call any method with image paths, PIL images, or numpy arrays directly
response = client.DetectSequence(
    images=["sample.jpg", "path/to/frame.png"],
    threshold=0.5
)

# 3. Response is automatically unwrapped into standard Python types!
annotated_images = response["images"]  # List of bytes
print("Received", len(annotated_images), "results")
```

## Package-Agnostic Image Inputs

You can pass image arguments in any format without converting them yourself:

```python
from pathlib import Path
from PIL import Image
import numpy as np

# File path
response = client.Process(images=Path("frame.jpg"))

# PIL Image
pil_img = Image.open("photo.png")
response = client.Process(images=pil_img)

# OpenCV / NumPy array
frame = np.zeros((480, 640, 3), dtype=np.uint8)
response = client.Process(images=frame)

# List of mixed formats
response = client.DetectSequence(images=["frame1.jpg", pil_img, frame])
```

## High-Level Box Wrappers

Pre-configured box wrappers are available for convenience:

```python
from blocks_sdk import YOLO, CoTracker, TAPNext, VGGT

yolo = YOLO("localhost:8062")
results = yolo.detect(images=["image1.jpg", "image2.jpg"], threshold=0.5)

cotracker = CoTracker("localhost:8063")
tracks = cotracker.track_video(video="video.mp4")

vggt = VGGT("localhost:8061")
reconstruction = vggt.reconstruct(images=["cam1.jpg", "cam2.jpg"])
```
