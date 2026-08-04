# TAPNext Tracker gRPC Service

A gRPC service for TAPNext point tracking with streaming support, following the vggt image pattern.

## Features

- Frame-by-frame tracking with state preservation
- Grid-based query point detection on first frame
- Server-side track accumulation until reset
- CUDA-accelerated inference using JAX/PyTorch backend
- Configurable grid size for query point density

## Directory Structure

```
tapnext_tracker/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto     # Shared proto definition
│   └── aux.py             # Helper functions for Value wrapping
├── src/
│   └── tapnext_service.py # Main service implementation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Docker with CUDA support (nvidia-docker2 or Docker with GPU support)
- TAPNext checkpoint file: `bootstapnext_ckpt.npz` (740MB)

Checkpoint download:
```bash
wget https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
```

## Building the Docker Image

Copy or mount the checkpoint into the build context before building:

```bash
cp /home/manuelf/tapnet/tapnet/tapnext/bootstapnext_ckpt.npz /path/to/tapnext_tracker/
docker build --tag sipgisr/tapnexttracker \
    --build-arg SERVICE_NAME=tapnext \
    -f docker/Dockerfile .
```

Or use volume mount at runtime:

```bash
docker run -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz \
    sipgisr/tapnexttracker
```

## Service Usage

### gRPC Protocol

The service uses the shared `PipelineService` interface with `Envelope` messages.

#### Request Configuration

Send a JSON config in the `config_json` field:

```json
{
  "tapnext": {
    "command": "track",
    "parameters": {
      "grid_size": 32,
      "reset": false
    }
  },
  "stream": 0
}
```

#### Tracking Request

```python
import grpc
import pipeline_pb2 as proto
from aux import wrap_value, unwrap_value

# Initialize gRPC client
channel = grpc.insecure_channel('localhost:8061')
stub = proto.PipelineServiceStub(channel)

# Prepare frame data
with open('frame.jpg', 'rb') as f:
    frame_bytes = f.read()

request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {
            "command": "track",
            "parameters": {"grid_size": 32, "reset": False}
        }
    }),
    data={"images": [wrap_value(frame_bytes)]}
)

response = stub.Process(request)
tracks = unwrap_value(response.data["tracks"])
visibles = unwrap_value(response.data["visibles"])
```

#### Reset Tracking State

```python
request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {
            "command": "reset"
        }
    })
)
response = stub.Process(request)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grid_size` | int | 32 | Number of grid points per dimension (grid_size × grid_size total) |
| `reset` | bool | false | Reset tracking state and start fresh |

### Response Format

```json
{
  "tapnext": {
    "status": "done",
    "runtime": 0.45,
    "frames_processed": 1
  }
}
```

Response data contains:
- `tracks`: Float32 array of shape (frame_count, num_points, sequence_len, 2)
- `visibles`: Float32 array of track visibility logits

## Docker Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `WORKSPACE` | /workspace | Workspace directory inside container |
| `SERVICE_NAME` | tapnext | Service identifier (affects proto and service file lookup) |

## Deployment

### Local Development

```bash
docker build --tag tapnexttracker -f docker/Dockerfile .
docker run --gpus all -p 8061:8061 tapnexttracker
```

### Production with GPU

```bash
docker run --gpus all \
    -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz \
    -e PORT=8061 \
    --name tapnext tracker \
    tapnexttracker
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8061 | Server listening port |
| `TORCH_HOME` | /workspace/.cache | PyTorch model cache directory |
| `HF_HOME` | /workspace/.cache | HuggingFace cache directory |

## Performance Notes

- First inference call loads the model (~740MB checkpoint)
- Model stays on GPU after loading unless idle for 60 seconds
- Grid detection generates (grid_size × grid_size) query points per frame
- Track state accumulates until explicitly reset

## Troubleshooting

### Checkpoint Not Found

Ensure checkpoint is at `/workspace/bootstapnext_ckpt.npz`:

```bash
docker run -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz tapnexttracker
```

### CUDA Out of Memory

Reduce batch size by processing frames individually or use smaller grid_size.
