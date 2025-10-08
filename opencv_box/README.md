# OpenCV Feature Service

The `opencv_box` package wraps several classic computer-vision utilities behind a gRPC service. The default implementation offers

- **Feature extraction & matching** via the `Process` RPC (SIFT or ORB) with FLANN/RANSAC post-processing when two images are supplied.
- **Frame similarity gating** via the `similarity_check` RPC using SSIM to decide whether a frame changed sufficiently from the previous one.

This README summarises the code layout, setup steps, request/response contracts, and Docker workflow.

---

## Directory Overview

| Path | Purpose |
| --- | --- |
| `src/opencv_service.py` | gRPC server exposing feature extraction/matching and similarity checking. |
| `protos/pipeline.proto` | Service definition shared with other boxes (`Envelope` message, RPC methods). |
| `protos/aux.py` | Helper utilities for wrapping/unwrapping protobuf `Value` maps. |
| `docker/` | Dockerfile for containerised deployment. |
| `test/` | Sample assets and a notebook (`test.ipynb`) for manual validation. |

---

## Prerequisites

- Python 3.10+
- OpenCV and scikit-image dependencies (installed via `requirements.txt`)
- Optional: GPU is **not** required; computation runs on CPU.

Install dependencies inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Service Locally

From the `opencv_box/` directory:

```bash
python src/opencv_service.py
```

Environment variables:
- `PORT` (default `8061`): gRPC listen port for the service.

Reflection metadata is enabled, so you can inspect the service via `grpcurl` or similar tools once it is running.

---

## RPC Endpoints

Both RPCs exchange messages using the generic `Envelope` schema. `config_json` carries command metadata, while `data` stores binary payloads encoded through `aux.wrap_value`.

### `Process`

- **Purpose**: extract keypoints/descriptors for each input image and, when two images are present, compute FLANN matches plus a RANSAC fundamental matrix.
- **Input**:
  - `config_json` example:
    ```json
    {
      "opencv": {
        "parameters": {
          "feature_extractor": "SIFT",
          "ratio_thresh": 0.75,
          "max_keypoints": 500
        }
      }
    }
    ```
  - `data["images"]`: list of raw image bytes (e.g., JPEG). Bytes are decoded with `cv2.imdecode`.
- **Output** (all tensors serialised with `numpy.save`):
  - `keypoints`: per-image array of `(x, y)` locations padded to a common shape.
  - `descriptors`: per-image descriptor matrix (SIFT 128-dim or ORB 32-dim) padded and stacked.
  - `matches_inliers_a` / `matches_inliers_b` (only when two images supplied): RANSAC inlier coordinates from image A/B.
  - `fundamental_matrix`: estimated 3×3 fundamental matrix (or empty array when matches insufficient).
  - `config_json` echoes status, runtime, and timestamp.

### `similarity_check`

- **Purpose**: lightweight frame-change detector for video streams.
- **Input**:
  - `config_json` example:
    ```json
    {
      "opencv": {
        "parameters": {
          "ssim_thresh": 0.92,
          "blur_kernel": 5
        }
      }
    }
    ```
  - `data["images"]`: expects the latest frame (JPEG bytes). The service keeps an internal previous frame per instance.
- **Output**:
  - `config_json` reports `ssim`, `changed` flag, and runtime. On the first request the service always returns `changed = true`.
  - `data["images"]`: includes the original frame bytes only if `changed` is true (useful for downstream publishing).

---

## Client Helpers

The protobuf helper functions in `protos/aux.py` convert between Python lists/bytes and the `Value` message used in the gRPC payload. Typical usage in a client:

```python
from protos import pipeline_pb2, pipeline_pb2_grpc, aux
import grpc, json

channel = grpc.insecure_channel("localhost:8061")
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

with open("image0.jpg", "rb") as f:
    img_bytes = f.read()

request = pipeline_pb2.Envelope(
    config_json=json.dumps({"opencv": {"parameters": {"feature_extractor": "SIFT"}}}),
    data={"images": aux.wrap_value([img_bytes])}
)
response = stub.Process(request)
```

---

## Docker Workflow

Build the container (from the repository root):

```bash
docker build \
  --tag sipgisr/opencvbox \
  -f opencv_box/docker/Dockerfile .
```

Run the container:

```bash
docker run --rm -it \
  -p 8061:8061 \
  sipgisr/opencvbox
```

---

## Testing

- `test/test.ipynb` demonstrates how to send requests and visualise matches.
- Sample pairs `test/00.jpg` and `test/01.jpg` help validate matching pipelines quickly.
- Enable logging (`logging.basicConfig(level=logging.INFO)`) for additional diagnostics when running locally.

---

## Extensibility

The service structure makes it straightforward to add new OpenCV routines: implement the logic inside `PipelineService`, register a new RPC in `pipeline.proto`, regenerate the stubs, and update any clients accordingly.
