# Usage Examples

## Quick Start

```python
from blocks_sdk import Client

client = Client("localhost:8062")

# Pass image paths, PIL images, NumPy arrays, or bytes directly!
response = client.DetectSequence(images=["image1.jpg", "image2.png"], threshold=0.5)

# Automatic unwrapping into Python native dict:
results = response["images"]
print(f"Received {len(results)} annotated image frames")
```

## YOLO Service

Detect objects in images:

```python
from blocks_sdk import YOLO

yolo = YOLO("localhost:8062")

# Pass file paths directly without manual open() / read()
response = yolo.detect(["image.jpg"], threshold=0.5)

# Results are already unwrapped as a list of bytes
annotated_images = response["images"]
```

Track objects across frames:

```python
response = yolo.track(["frame1.jpg", "frame2.jpg", "frame3.jpg"], stream_id=0)
results = response["images"]
```

## TAPNext Service

Track points in video:

```python
from blocks_sdk import TAPNext

tapnext = TAPNext("localhost:8061")

response = tapnext.track(["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"], grid_size=32)

# Direct access to unwrapped output fields
tracks = response.get("tracks", [])
visibles = response.get("visibles", [])
```

Reset tracking state:

```python
tapnext.reset()
```

## Low-Level Helper Usage

```python
from blocks_sdk import to_bytes, wrap_envelope, unwrap_envelope

# Convert PIL Image or filepath to bytes package-agnostically
raw_bytes = to_bytes("image.jpg")

# Build Envelope Protobuf message
envelope = wrap_envelope(data={"images": [raw_bytes]}, config={"threshold": 0.5})

# Unwrap Envelope Protobuf response to Python dict
result_dict = unwrap_envelope(envelope)
```
