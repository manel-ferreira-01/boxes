# Usage Examples

## Quick Start

```bash
# Build and start all services
docker compose up -d --build

# Check running containers  
docker compose ps

# Test from host machine
python3 test_client.py
```

## Services Configuration

The docker-compose.yml includes build args for each service:

| Service | Host Port | Container Name |
|---------|-----------|----------------|
| TAPNext (GPU) | 8061 | tapnext-service |
| YOLO | 8062 | yolo-service |
| CoTracker | 8063 | cotracker-service |
| TextEmbedding | 8064 | text-embedding-service |

## Python API

### Import and Initialize

```python
from blocks_sdk import YOLO, TAPNext, CoTracker, TextEmbedding

# From host machine (with port mapping)
yolo = YOLO("localhost:8062")
tapnext = TAPNext("localhost:8061")

# From another container on same network
cotracker = CoTracker("cotracker-service:8063")
text_embed = TextEmbedding("text-embedding-service:8064")
```

### YOLO Service

```python
from blocks_sdk import YOLO

yolo = YOLO("localhost:8062")

# Detect objects in images
with open("image.jpg", "rb") as f:
    img_data = f.read()

response = yolo.detect([img_data], threshold=0.5)

# Track across video frames
frames = [frame1, frame2, frame3]
tracks = yolo.track(frames, stream_id=0)
```

### TAPNext Service (GPU)

```python
from blocks_sdk import TAPNext

tracker = TAPNext("localhost:8061")

# Track points in video frames
video_frames = [frame1_bytes, frame2_bytes, ...]
tracks = tracker.track(video_frames, grid_size=32)

# Reset tracking state
tracker.reset()
```

### CoTracker Service (GPU)

```python
from blocks_sdk import CoTracker

cotracker = CoTracker("localhost:8063")

# Track motion in video
video_frames = [...]  # List of image bytes
tracks = cotracker.track_video(video_frames)
```

### TextEmbedding Service (GPU)

```python
from blocks_sdk import TextEmbedding

embedder = TextEmbedding("localhost:8064")

# Encode text to embeddings
texts = ["hello world", "test sentence"]
embeddings = embedder.encode(texts)

# Check similarity between sentences
similarity = embedder.similarity("hello", "hi there")
```

## Error Handling

```python
from blocks_sdk import YOLO
import grpc

yolo = YOLO("localhost:8062")

try:
    response = yolo.detect([img_data])
except grpc.RpcError as e:
    print(f"gRPC error: {e.code()}")
    print(f"Details: {e.details()}")
except Exception as e:
    print(f"Other error: {e}")

# Cleanup
yolo.close()
```

## Debugging

```bash
# View service logs
docker compose logs -f tapnext-service

# Check if service is running
docker compose ps

# Rebuild a specific service
docker compose build yolo-service

# Stop all services
docker compose down
```
