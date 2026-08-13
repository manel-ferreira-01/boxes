# Quick Start Guide

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ (for local development/testing)
- Basic understanding of gRPC concepts helpful but not required

## Your Existing Setup

### Images Directory (`/images/`)

Each subdirectory is a complete AI service:

```
/images/
├── opencv_box/         # Feature matching, optical flow
├── vggt/              # 3D reconstruction  
├── yologpt/           # YOLO detection & tracking
├── cotracker/         # Video motion tracking
├── gradio_display/    # Web UI
└── textEmbedding/     # Text embeddings
```

### Pipelines Directory (`/pipelines/`)

Pre-configured pipeline examples:

```
/pipelines/
├── yolo/               # Simple YOLO detection
├── vggt/               # 3D reconstruction pipeline
├── gradio+vggt+yolo/   # Full multi-algorithm pipeline
└── folder_wd_yolo/     # File-watcher + YOLO
```

## Run a Pipeline (Using Existing Images)

### Step 1: Pull Images (if not built locally)

```bash
docker pull sipgisr/displaygrpc
docker pull sipgisr/yologrpc  
docker pull sipgisr/vggtgrpc
docker pull sipgisr/maestro:v1-latest
```

### Step 2: Choose a Pipeline

For example, the simplest YOLO pipeline:

```yaml
# /home/manuelf/boxes/pipelines/yolo/config.yaml
kind: pipeline
spec:
  name: Yolo
---
kind: stage
spec:
  name: yolo_detect
  method: AllProcessing
  address: yologrpc:8061
  pipeline: Yolo
---
kind: stage
spec:
  name: gradio-source
  method: acquire
  address: interface:8061
  pipeline: Yolo
---
kind: stage
spec:
  name: gradio-display
  method: display
  address: interface:8061
  pipeline: Yolo
---
# (links continue...)
```

### Step 3: Run with Docker Compose

```bash
cd /home/manuelf/boxes/pipelines/yolo

docker-compose up -d
```

### Step 4: Access the Interface

- Open browser to `http://localhost:7860`
- Upload images for detection
- See results annotated with bounding boxes

## Build a Custom Image (For Your Own AI Service)

### Scenario: Create a new face detection service

#### Step 1: Create Directory Structure

```bash
mkdir -p /home/manuelf/boxes/images/face_detector/{protos,src,docker}
```

#### Step 2: Write Protobuf Definition

Create `/home/manuelf/boxes/images/face_detector/protos/pipeline.proto`:

```protobuf
syntax = "proto3";

package pipeline;

message Envelope {
  string config_json = 1;
  map<string, Value> data = 2;
}

message Value {
  oneof kind { 
    bytes b       = 1;
    string s      = 2;
    float f       = 5;
  }
}

service PipelineService {
  rpc DetectFaces(Envelope) returns (Envelope);
}
```

#### Step 3: Generate gRPC Code

```bash
cd /home/manuelf/boxes/images/face_detector/protos
python -m grpc_tools.protoc \
    --python_out=. \
    --grpc_python_out=. \
    pipeline.proto
```

#### Step 4: Write the Service

Create `/home/manuelf/boxes/images/face_detector/src/face_service.py`:

```python
import sys
sys.path.append('./protos')
import pipeline_pb2 as pb2
import pipeline_pb2_grpc as pb2_grpc
from aux import wrap_value, unwrap_value

class FaceService(pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        # Load face detection model
        self.model = load_face_detector_model()
    
    def DetectFaces(self, request, context):
        images = unwrap_value(request.data.get("images", []))
        
        results = []
        for img_bytes in images:
            faces = self.model.detect(img_bytes)
            results.append(faces)
        
        return pb2.Envelope(
            config_json='{"status": "success"}',
            data={"faces": wrap_value(results)}
        )
```

Create `/home/manuelf/boxes/images/face_detector/src/aux.py`:

```python
import pipeline_pb2

def wrap_value(obj):
    if isinstance(obj, bytes):
        return pipeline_pb2.Value(b=obj)
    elif isinstance(obj, str):
        return pipeline_pb2.Value(s=obj)
    return pipeline_pb2.Value(b=b"")

def unwrap_value(val):
    kind = val.WhichOneof("kind")
    if kind == "b":
        return val.b
    elif kind == "s":
        return val.s
    return None
```

#### Step 5: Create Dockerfile

Create `/home/manuelf/boxes/images/face_detector/docker/Dockerfile`:

```dockerfile
ARG SERVICE_NAME=face_service

FROM python:3.10-slim AS builder
RUN pip install grpcio grpcio-tools protobuf
COPY protos /workspace/
WORKDIR /workspace
RUN python -m grpc_tools.protoc --python_out=. --grpc_python_out=. pipeline.proto

FROM python:3.10-slim
ARG USER=runner
RUN addgroup --system runner-group && \
    adduser --system --ingroup runner-group runner && \
    mkdir /workspace && chown runner:runner /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --from=builder /workspace/*.py /workspace/
COPY src/face_service.py /workspace/service.py
COPY protos/pipeline.proto /

USER runner
EXPOSE 8061
CMD ["python", "/workspace/service.py"]
```

Create `/home/manuelf/boxes/images/face_detector/requirements.txt`:

```txt
grpcio
protobuf
grpcio-reflection
grpcio-status
numpy
cv2 face detection library here
```

#### Step 6: Build and Run

```bash
cd /home/manuelf/boxes/images/face_detector
docker build -t my_face_detector -f docker/Dockerfile .
docker run --rm -it -p 8061:8061 my_face_detector
```

## Common Service Types Reference

### CPU-only Services (No GPU)

Best for: Small models, preprocessing, simple logic

**Image base:** `python:3.10-slim`
- opencv_box
- folder_wd
- textEmbedding (CPU mode)

### GPU-accelerated Services

Best for: Heavy AI/ML inference

**Image base:** `nvidia/cuda:12.2.2-base-ubuntu22.04`
- vggt
- cotracker  
- yologpt

## Pipeline Pattern Reference

See `/home/manuelf/boxes/docs/Pipeline_Configuration_Reference.md` for complete patterns.

### Simple Sequential Pipeline

```
Input → Process Output
```

### Flow-Controlled Pipeline

```
Input → Check if change → Process only if changed → Output
```

### Parallel Processing Pipeline

```
                    ┌──▶ Algorithm A
Input ──┬──▶ Split ├──▶ Algorithm B
        └────────────▶ Algorithm C
```

## Next Steps

1. **Explore existing pipelines:** Read config files in `/home/manuelf/boxes/pipelines/`
2. **Test services standalone:** Use Python gRPC client to test individual services
3. **Build custom service:** Follow the quick start above for your use case
4. **Create pipeline config:** Define how services connect in Maestro

## Troubleshooting

### Service not responding on port 8061

- Check container logs: `docker logs <container>`
- Verify service is running: `docker exec -it <container> ps aux | grep python`

### Pipeline fails with "stage not found"

- Check that all stage names in links match exactly (case-sensitive)
- Ensure stages reference the same pipeline name

### Images not loading (CUDA OOM)

- Reduce batch size
- Add idle timeout to move model back to CPU: `IDLE_TIMEOUT = 60`

## Help

See other docs:
- `/home/manuelf/boxes/docs/Pipeline_Architecture_Overview.md`
- `/home/manuelf/boxes/docs/gRPC_Services_Reference.md`
- `/home/manuelf/boxes/docs/Pipeline_Configuration_Reference.md`
