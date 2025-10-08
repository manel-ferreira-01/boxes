# Gradio Display Service

The `gradio_display` package provides a Gradio-based front end and companion gRPC service that orchestrate computer-vision pipelines. Current examples ship with YOLO detection/tracking and VGGT 3D reconstruction, but the contract is designed to accommodate additional algorithms as you expand the toolbox. The service sits between the web UI and algorithm backends, moving media and metadata across components via protobuf messages and lightweight file handoffs.

This README walks through the project layout, local setup, runtime configuration, and the gRPC contract used to interact with the display component.

---

## Directory Overview

| Path | Purpose |
| --- | --- |
| `src/display_service.py` | gRPC server exposing acquire/display endpoints and bridging to the UI. |
| `src/display_interface.py` | Gradio `Blocks` application with example tabs (YOLO detection/tracking, VGGT reconstruction) that can be extended with new algorithms. |
| `src/utils.py` | Helpers for serializing outputs, resizing media, and writing temporary artifacts. |
| `protos/pipeline.proto` | Defines the `DisplayService` RPC interface and `Envelope` message. |
| `docker/` | Dockerfiles and instructions for containerizing the service. |
| `test/test_display_client.ipynb` | Notebook demonstrating how to drive the service programmatically. |

---

## Prerequisites

- Python 3.10+
- (Optional) CUDA-capable GPU if the downstream YOLO/VGGT services require it
- System packages used by OpenCV (`ffmpeg`, `libsm6`, `libxext6`) when running outside Docker

Install the Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Service Locally

1. Ensure the collaborating services are reachable (e.g., the bundled YOLO and VGGT pipelines, plus any additional backends you register).
2. From the `gradio_display/` directory start the server:
   ```bash
   python src/display_service.py
   ```
3. Browse to the Gradio UI (defaults to `http://localhost:7860`).

Environment variables:
- `PORT` (default `8061`): gRPC listen port for `DisplayService`
- Gradio launches on `0.0.0.0:7860` with public sharing enabled by default (see `launch` arguments in `display_interface.py`).

---

## gRPC Endpoints

The service uses a generic `Envelope` message consisting of a JSON control payload (`config_json`) and a map of typed `Value` entries (`data`). The protobuf definition lives in `protos/pipeline.proto`.

### `acquire`
- Non-blocking poll invoked by algorithm workers to fetch pending UI jobs.
- Returns `images` (JPEG-encoded bytes) along with metadata in `config_json`. For the bundled examples, the payload looks like:
  ```json
  {
    "aispgradio": {
      "command": "detectsequence" | "tracksequence" | "3d_infer",
      "user": "<gradio session label>",
      "input_count": 1,
      "timestamp": "<ISO8601>",
      "parameters": { ... }
    }
  }
  ```
- When no work is available, the response includes `{ "aispgradio": { "empty": "empty" } }` and a placeholder frame.

### `display_yolo`
- Accepts annotated image sequences produced by YOLO pipelines (or other detectors you integrate under the same contract).
- Expects `images` data (list of JPEG bytes) and a matching `config_json` echoing the original request.
- Writes results to disk so the Gradio UI can surface detection or tracking overlays, CSV/JSON exports, and preview videos.

### `display_vggt`
- Accepts VGGT reconstruction outputs (or future 3D algorithms you plug in), typically providing the binary `.glb` model in `data["glb_file"]`. 
- Once the mesh is persisted, the UI updates the 3D viewer tab for the requesting session.

---

## Data Flow

1. Users interact with Gradio tabs (YOLO, VGGT, or any custom additions) to submit image galleries or videos.
2. The UI writes serialized requests to per-algorithm files under the configured temporary directory (default `/tmp`).
3. Background workers call `acquire` to collect jobs, run the heavy computation, and respond via `display_yolo` or `display_vggt`.
4. The UI watches for output files and updates galleries, videos, downloads, and 3D viewers accordingly.

This simple file-based queue keeps the UI responsive while isolating heavy processing in external services.

---

## Docker Workflow

Prebuilt Dockerfiles under `docker/` simplify deployment. The image exposes both the gRPC port (`8061`) and Gradio port (`7860`).

Build the image from the repository root:
```bash
docker build \
  --tag sipgisr/gradiosipg \
  -f gradio_display/docker/Dockerfile .
```

Run the container:
```bash
docker run --rm -it --gpus all \
  -p 8061:8061 -p 7860:7860 \
  sipgisr/gradiosipg
```

Open `http://localhost:7860` to access the UI.

---

## Testing & Development

- Use `test/test_display_client.ipynb` to simulate end-to-end requests and verify the gRPC contract.
- For iterative UI changes, `Dockerfile.dev` provides a lighter image with live-reload tooling.
- Log output is configured via Python's standard `logging` module within `display_service.py`.

---

## Troubleshooting

- **No jobs picked up**: Confirm the worker calling `acquire` runs with correct path permissions and that the Gradio session generated inputs.
- **Large file handling**: The gRPC server lifts send/receive limits (`-1`) to allow sizeable image batches and meshes; ensure clients mirror these limits.
- **Port conflicts**: Override `PORT` or adjust the Gradio `launch` parameters if 8061/7860 are already in use.

---

## Related Components

- YOLO pipeline (`folder_wd/`, `opencv_box/`) for detection/tracking backends.
- VGGT pipeline (`vggt/`) for 3D reconstruction.
- Future algorithm directories you add can follow the same pattern.
- Shared protobuf helpers in `protos/aux.py` for wrapping/unwrapping repeated values.
