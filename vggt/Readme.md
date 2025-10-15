# VGGT 3D Reconstruction Service

The `vggt` package bundles Meta's **VGGT** model into a gRPC microservice that performs dense 3D reconstruction from a short sequence of RGB images. The service exposes a single `PipelineService.Process` RPC that ingests raw image bytes, runs VGGT inference, and returns depth, point cloud data, camera intrinsics/extrinsics, and a ready-to-visualize `.glb` mesh.

This README explains how to set up the environment, launch the service (locally or via Docker), craft requests, and interpret responses.

---

## Table of Contents
1. [Repository Layout](#repository-layout)
2. [Prerequisites](#prerequisites)
3. [Model Weights](#model-weights)
4. [Local Setup](#local-setup)
5. [Running the gRPC Service](#running-the-grpc-service)
6. [gRPC API](#grpc-api)
7. [Outputs](#outputs)
8. [Docker Workflow](#docker-workflow)
9. [Testing and Notebooks](#testing-and-notebooks)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Repository Layout

Key directories in `vggt/`:

| Path | Description |
| --- | --- |
| `src/vggt_service.py` | gRPC server implementation that wraps VGGT inference. |
| `src/utils/preprocess.py` | Image preprocessing utilities (resize, crop/pad, batching). |
| `src/vggt/` | Vendor copy of the official VGGT codebase for inference utilities. |
| `protos/pipeline.proto` | Common `Envelope` request/response contract used by the service. |
| `docker/` | Dockerfile and cached `vggt-1b.pt` weights for building GPU-enabled container images. |
| `test/` | Jupyter notebook demonstrating end-to-end requests against a running service. |

---

## Prerequisites

- **Python**: 3.10 or newer is recommended.
- **PyTorch**: GPU support is optional but strongly recommended (VGGT is large). CPU inference works for experimentation but is slow.
- **System packages**: `ffmpeg`, `libsm6`, and `libxext6` are required for OpenCV (installed automatically in Docker build).
- **CUDA Toolkit** (optional): Needed for GPU acceleration when running outside Docker.

The Python dependencies are listed in `requirements.txt`. Install them once the virtual environment is active:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Model Weights

The service expects the VGGT 1B checkpoint to be available at the project root as `vggt-1b.pt`. There are two typical ways to obtain it:

1. **Download from Hugging Face** (recommended):
   ```bash
   wget https://huggingface.co/facebook/VGGT-1B/resolve/main/vggt-1b.pt -O vggt-1b.pt
   ```

2. **Re-use the cached artifact** bundled in `docker/vggt-1b.pt` when building the Docker image (see [Docker Workflow](#docker-workflow)).

Keep the file alongside `src/` so that `vggt_service.py` can load it when the process starts.

---

## Local Setup

1. Create and activate a virtual environment (example with `venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install service dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `vggt-1b.pt` is present at the root of the package (see previous section).

Optional: set `TORCH_HOME`, `HF_HOME`, and `HF_HUB_CACHE` to reuse model caches between runs (Docker sets them automatically).

---

## Running the gRPC Service

From the `vggt/` directory:

```bash
python src/vggt_service.py
```

Environment variables:
- `PORT` (optional): Port to bind (defaults to `8061`).

The server exposes reflection metadata, so tools like `grpcurl` can introspect the service once it is running.

---

## gRPC API

### Service definition

`protos/pipeline.proto` defines a generic `PipelineService` with a single method:

```proto
service PipelineService {
  rpc Process(Envelope) returns (Envelope);
}
```

The `Envelope` contains a `config_json` string and a map of typed `Value`s (`bytes`, `floats`, or repeated variants).

### Request contract

- **`config_json`**: JSON string describing the command:
  ```json
  {
    "aispgradio": {
      "command": "3d_infer",
      "parameters": {
        "conf_threshold": 30
      }
    }
  }
  ```
  - `command` must be `"3d_infer"` for VGGT inference.
  - `conf_threshold` (optional) controls the mesh confidence threshold. Defaults to `30` if omitted.
  - Passing `{"aispgradio": {"empty": "empty"}}` results in an empty response.

- **`data["images"]`**: List of raw image bytes. All images must have identical height/width and be RGB. The service converts them to tensors, normalizes to `[0, 1]`, and applies the preprocessing pipeline in `src/utils/preprocess.py`.

### Python client example

```python
import grpc
import json
from protos import pipeline_pb2, pipeline_pb2_grpc, aux


def load_image_bytes(paths):
    return [open(p, "rb").read() for p in paths]


def make_request(image_paths):
    config = {
        "aispgradio": {
            "command": "3d_infer",
            "parameters": {"conf_threshold": 25, "device": "cuda:0"}
        }
    }
    images = load_image_bytes(image_paths)
    return pipeline_pb2.Envelope(
        config_json=json.dumps(config),
        data={"images": aux.wrap_value(images)}
    )


channel = grpc.insecure_channel("localhost:8061")
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
response = stub.Process(make_request(["frame_000.png", "frame_001.png"]))
```

---

## Outputs

Successful `Process` calls return an `Envelope` whose `config_json` echoes the command and whose `data` map contains:

| Key | Type | Description |
| --- | --- | --- |
| `world_points` | serialized PyTorch tensor | Dense point cloud in world coordinates. |
| `world_points_conf` | serialized PyTorch tensor | Confidence scores for each world point. |
| `depth` | serialized PyTorch tensor | Depth map per input view. |
| `depth_conf` | serialized PyTorch tensor | Confidence mask aligned with `depth`. |
| `extrinsic` | serialized PyTorch tensor | Estimated camera extrinsics. |
| `intrinsic` | serialized PyTorch tensor | Estimated camera intrinsics. |
| `images` | serialized NumPy array | Preprocessed images fed to VGGT (float, CHW). |
| `glb_file` | bytes | Binary `.glb` scene generated via `predictions_to_glb`. |

The tensors are serialized using `torch.save`. Use `torch.load(io.BytesIO(...))` to recover them on the client side. `glb_file` is ready to store on disk or stream to a viewer.

```python
import io
import torch
from protos import pipeline_pb2, pipeline_pb2_grpc, aux

# assume `response` is a pipeline_pb2.Envelope returned by PipelineService.Process
world_points = torch.load(io.BytesIO(aux.unwrap_value(response.data["world_points"])))
depth = torch.load(io.BytesIO(aux.unwrap_value(response.data["depth"])))
extrinsic = torch.load(io.BytesIO(aux.unwrap_value(response.data["extrinsic"])))

with open("scene.glb", "wb") as f:
    f.write(aux.unwrap_value(response.data["glb_file"]))
```

---

## Docker Workflow

1. Build the image from the repository root (the `SERVICE_NAME` argument controls the server entry point copied into the image):
   ```bash
   docker build \
     --tag sipgisr/vggtgrpc \
     --build-arg SERVICE_NAME=vggt \
     -f docker/Dockerfile .
   ```
   The Docker build installs dependencies, copies `docker/vggt-1b.pt` to the workspace, and exposes port `8061`.

2. Run the container with GPU access:
   ```bash
   docker run --rm -it --gpus all \
     -p 8061:8061 \
     sipgisr/vggtgrpc \
     bash -c "python3 service.py"
   ```

   Add a bind mount if you need to persist the model cache:
   ```bash
   docker run --rm -it --gpus all \
     -p 8061:8061 \
     --mount type=bind,source=$PWD/.cache,target=/workspace/.cache \
     sipgisr/vggtgrpc \
     bash -c "python3 service.py"
   ```

---

## Testing and Notebooks

- `test/test_vggt.ipynb`: Interactive notebook that demonstrates request assembly and response parsing. Update the host/port configuration before running.
- `test/images/`: Sample imagery for quick experiments.

---

## Troubleshooting

- **Mismatched image sizes**: The server validates that every image in the request has the same shape and raises an explicit error otherwise.
- **Slow inference / CPU fallback**: Ensure CUDA is available. The service keeps the model on CPU when idle; the first request after a cold start may take longer.
- **Large responses**: The `.glb` payload can reach several MB. The server sets gRPC message limits to `-1`, but clients may need to increase their receive limits.
- **Cache reuse**: Set `TORCH_HOME`/`HF_HOME` to avoid repeated downloads when running outside Docker.

---

## References

- VGGT paper and official repository: <https://github.com/facebookresearch/vggt>
- gRPC Python basics: <https://grpc.io/docs/languages/python/basics/>
