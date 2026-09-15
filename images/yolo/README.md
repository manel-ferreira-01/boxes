# YOLO Box (object detection on images and videos)

A gRPC box that runs YOLO object detection (ultralytics) over **images and
videos** and returns per-frame detection records plus optionally annotated
frames.

The box speaks the shared **envelope** interface, so it is addressable
through `boxes_client` exactly like clip:

```python
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

It is **stateless**: one call = one detection pass over every frame you
send (images and/or one decoded video). `reset` is accepted as a no-op so
`boxes_client`'s `reset_first` stays safe.

## Directory structure

```
yolo/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── yolo_service.py        # PipelineService.Process(Envelope)
├── test/
│   ├── test_yolo.py           # in-process / remote smoke test (dummy images)
│   ├── dog.jpg                # test fixture
│   └── car.jpg                # test fixture
├── requirements.txt
└── README.md
```

## Build

```bash
docker build --tag sipgisr/yolo --build-arg SERVICE_NAME=yolo -f docker/Dockerfile .
```

The default checkpoint (`yolov8n.pt`, ~6 MB) is **baked into the image at
build time** and cached under `$HOME`, so the first call never downloads.

## Run

```bash
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/yolo
```

## Service usage

### Request

Up to two fields in `data` (send **either**, not both), and a small `config`:

- `data["images"]` — list of image bytes (JPEG/PNG/…), one detection record
  per image.
- `data["video"]` — a single video file (mp4/avi/webm/mov). The box decodes
  it server-side (OpenCV/ffmpeg), samples every `frame_step`-th frame, and
  stops at `max_frames` sampled frames.
- `config["yolo"]["command"]` — `"detect"` (default) or `"reset"` (no-op,
  the box is stateless).
- `config["yolo"]["parameters"]` — the knobs:

| key               | default | meaning                                                        |
|-------------------|---------|----------------------------------------------------------------|
| `weights`         | —       | accepted for API consistency; the box runs the checkpoint loaded at startup (`yolov8n.pt`) |
| `conf`            | `0.25`  | confidence threshold                                            |
| `iou`             | `0.70`  | NMS IoU threshold                                               |
| `imgsz`           | `640`   | inference image size                                            |
| `max_det`         | `300`   | max detections per frame                                        |
| `classes`         | all     | list of class ids to keep (e.g. `[16]` = dogs only)             |
| `device`          | auto    | `"cpu"` / `"cuda"` / `"cuda:N"` — wins over the auto lifecycle |
| `save_annotated`  | `true`  | also return the annotated frames as JPEGs                       |
| `frame_step`      | `1`     | video only: sample every Nth frame                              |
| `max_frames`      | `1024`  | safety cap on sampled frames per call                           |

### Response

`config_json` carries the status:

```json
{
  "yolo": {
    "status": "done",
    "runtime": 4.21,
    "source": "images",            // or "video"
    "num_frames": 2,
    "frames_sampled": 2,
    "num_detections": 3,
    "frames_in_video": 211,        // video only
    "frame_step": 30,              // video only
    "encoding": { "detections": "json", "annotated": "identity" }
  }
}
```

and `data` carries:

| field        | kind | description |
|--------------|------|-------------|
| `detections` | `b` (declared `json`) | list with **one record per frame**: see below |
| `annotated`  | `bb` (declared `identity`) | JPEG-annotated frame per input frame (only when `save_annotated`) |

A detection record (plain JSON):

```json
{
  "frame_index": 30,
  "width": 1920,
  "height": 1080,
  "boxes":       [[x1, y1, x2, y2], ...],   // xyxy in input-frame pixels
  "class_ids":   [49, ...],
  "labels":      ["orange", ...],
  "scores":      [0.65, ...]
}
```

`frame_index` is the frame's original index in the source (its position in
`images`, or the video frame number when sampling a video). `boxes` are
axis-aligned `xyxy` rectangles in the **original frame's** pixel
coordinates (rounded to 2 decimals).

## Call with boxes_client

```python
from boxes_client import Box
import pathlib

b = Box("localhost:8061")

# --- images ---------------------------------------------------------
res = b.run(
    data   = {"images": [pathlib.Path("dog.jpg"), pathlib.Path("car.jpg")]},
    config = {"yolo": {"command": "detect",
                       "parameters": {"conf": 0.25}}},
)
print(res.encoding)     # {'detections': 'json', 'annotated': 'identity'}
for d in res.detections:                       # decoded: a Python list
    for box, label, score in zip(d["boxes"], d["labels"], d["scores"]):
        print(label, score, box)
open("frame0.jpg", "wb").write(res.annotated[0])   # optional annotated JPEGs

# --- video ----------------------------------------------------------
res = b.run(
    data   = {"video": pathlib.Path("cozinha.mp4")},
    config = {"yolo": {"command": "detect",
                       "parameters": {"frame_step": 30, "max_frames": 16,
                                      "save_annotated": False}}},
)
print(res.config["yolo"])   # frames_sampled / frames_in_video / num_detections
```

Or use the vendored protos directly:

```python
import grpc, json, pipeline_pb2, pipeline_pb2_grpc, aux

channel = grpc.insecure_channel("localhost:8061",
                                options=[("grpc.max_send_message_length", -1),
                                         ("grpc.max_receive_message_length", -1)])
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

req = pipeline_pb2.Envelope(
    config_json=json.dumps({"yolo": {"command": "detect"}}),
    data={"images": aux.wrap_value([open("car.jpg", "rb").read()])},
)
resp = stub.Process(req)
dets = json.loads(aux.unwrap_value(resp.data["detections"]))  # bytes -> JSON
print(json.loads(resp.config_json), dets)
```

## GPU behaviour

The fleet convention: the model loads on **CPU at startup** (fast start, no
VRAM), moves to CUDA **in place** on the first request (unless
`parameters.device` says otherwise), and a watchdog thread moves it back to
CPU after ~60 s idle. `parameters.device` always wins.

## Test

A clip-style script (no notebook) that builds Envelopes with the bundled
test images, calls `Process`, and checks the decoded detection records, the
annotated JPEGs, the `reset` no-op, and — when a video is available — the
video input path:

```bash
# server already running at :8061
python test/test_yolo.py
BOX_HOST=10.0.0.5:8061 python test/test_yolo.py
YOLO_TEST_VIDEO=cozinha.mp4 python test/test_yolo.py
```

The video case is **skipped** (not failed) when no video file is available:
it checks `YOLO_TEST_VIDEO`, then the repo's `cozinha.mp4` at the repo root.
