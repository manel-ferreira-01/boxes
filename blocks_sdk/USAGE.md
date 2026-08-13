# Usage Examples

## Quick Start

```bash
# Ensure services are running
docker compose up -d

# Run an example script
python3 examples/simple_example.py
```

## Initialize Clients

Services run on these ports (from docker-compose.yml):
- TAPNext: `localhost:8061`
- YOLO: `localhost:8062`
- CoTracker: `localhost:8063`

```python
from blocks_sdk import TAPNext, YOLO, CoTracker

tapnext = TAPNext("localhost:8061")
yolo = YOLO("localhost:8062")
cotracker = CoTracker("localhost:8063")
```

##YOLO Service

Detect objects in images:

```python
from blocks_sdk import YOLO
import cv2
import numpy as np

yolo = YOLO("localhost:8061")

# Load image as bytes
with open("image.jpg", "rb") as f:
    img_bytes = f.read()

response = yolo.detect([img_bytes], threshold=0.5)

# Get annotated images (bytes)
from blocks_sdk import helpers
annotated_images = helpers.unwrap_value(response.data.get("images", []))

# Save results
for i, annotated_img in enumerate(annotated_images):
    nparr = np.frombuffer(annotated_img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(f"result_{i}.jpg", img)
```

Track objects across frames:

```python
# Load multiple frames
frames = []
for frame_file in ["frame1.jpg", "frame2.jpg", "frame3.jpg"]:
    with open(frame_file, "rb") as f:
        frames.append(f.read())

response = yolo.track(frames, stream_id=0)
results = helpers.unwrap_value(response.data.get("images", []))
```

## TAPNext Service

Track points in video:

```python
from blocks_sdk import TAPNext

tapnext = TAPNext("localhost:8061")

# Load frames as bytes
frames = []
for i in range(5):  # Track first 5 frames
    with open(f"frame_{i}.jpg", "rb") as f:
        frames.append(f.read())

response = tapnext.track(frames, grid_size=32)

# Get tracks and visibles
from blocks_sdk import helpers
tracks = helpers.unwrap_value(response.data.get("tracks", []))
visibles = helpers.unwrap_value(response.data.get("visibles", []))

print(f"Tracked {len(tracks)} frames with {len(visibles[0]) if visibles else 0} points")
```

Reset tracking state:

```python
tapnext.reset()
```
