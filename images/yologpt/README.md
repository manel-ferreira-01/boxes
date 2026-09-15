# YOLOv11 Box (object detection & multi-object tracking)

A gRPC box that runs **YOLOv11** ([Ultralytics](https://github.com/ultralytics/ultralytics))
for object **detection** and, with tracker state, **multi-object tracking**.

The box speaks the shared **envelope** interface, so it is addressable through
`boxes_client` (and the webui) exactly like clip / tapnext / lang_sam:

```protobuf
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

One RPC, dispatched on the `"command"` in the `yolo` config section (the
tapnext pattern) — `detect`, `track`, or `reset`.

## Directory structure

```
yologpt/
├── docker/
│   └── Dockerfile
├── protos/
│   ├── pipeline.proto         # shared proto (same as every other box)
│   ├── pipeline_pb2.py        # generated
│   ├── pipeline_pb2_grpc.py   # generated
│   └── aux.py                 # wrap_value / unwrap_value helpers
├── src/
│   └── yoloservice.py         # PipelineService.Process(Envelope)
├── test/
│   ├── test_yolo.py           # live test against a running box
│   └── test_yolo_fake.py      # in-process, fake model (no GPU)
└── README.md
```

## Build

```bash
docker build --tag sipgisr/yolov11 -f docker/Dockerfile .
```

The model weights (`yolo11n.pt`) are pulled from the Ultralytics release
assets at build time; the `yolo11n.pt` file can also be mounted at runtime
(see Run).

## Run

```bash
# GPU (recommended)
docker run --rm -p 8061:8061 --gpus all --ipc=host sipgisr/yolov11

# local weights instead of the baked-in ones:
docker run -p XXXX:8061 -v $(pwd)/yolo11n.pt:/workspace/yolo11n.pt sipgisr/yolov11
```

The box loads YOLOv11 at startup; first request is slightly slower (CUDA
warm-up). The tracker state is **server-global** (not per-session).

## Service usage

### Request

- `data["images"]` — list of image bytes (JPEG/PNG); a single image is a
  one-element list.
- `config["yolo"]["command"]` — `"detect"` (stateless) or `"track"`
  (stateful, ids accumulate across calls), or `"reset"` to clear the tracker.
- `config["yolo"]["parameters"]` — optional, forwarded to YOLOv11 as kwargs:
  - `conf` — confidence threshold (default `0.5`)
  - `iou` — IoU threshold (default `0.7`)
  - `tracker` — tracker backend for `track` mode (`"bytetrack.yaml"` default,
    or `"botsort.yaml"`)

```json
{ "yolo": { "command": "detect", "parameters": { "conf": 0.5 } } }
```

### Response

`config_json` carries the namespaced status + **declared payload encoding**,
and `data` carries two fields:

| field        | type                                          | description |
|--------------|-----------------------------------------------|-------------|
| `images`     | `bytes` per image (raw JPEG)                  | annotated image, boxes drawn, one per input |
| `detections` | `json`-encoded list of dicts                  | one FLAT, ordered list (see below) |

`detections` is a **flat** list — every row carries `image_index` so you can
group by input image. In `track` mode each row also has a `track_id`:

```json
[
  {"image_index": 0, "bbox": [x1, y1, x2, y2], "confidence": 0.89,
   "class_id": 0, "class_name": "person"},
  {"image_index": 0, "bbox": [..], "confidence": 0.52, "class_id": 2,
   "class_name": "car", "track_id": 3}
]
```

A row may carry `track_id: -1` — the tracker has not assigned that (new) track
an id yet (it first appears in a frame later than the sequence's first one;
its id shows up from the next frame). Established tracks keep their id across
calls; `detect` never touches the tracker state.

Status follows the fleet convention:
`"done"` · `"empty_request"` (no images) · `"error"` (with `"error"`).
`reset` answers `{"yolo": {"status": "done", "action": "reset"}}`.

## Call with boxes_client

```python
from boxes_client import Box

b = Box("localhost:8061")

# 1) detection (stateless)
res = b.run(
    data={"images": [pathlib.Path("dog.jpg"), pathlib.Path("car.jpg")]},
    config={"yolo": {"command": "detect", "parameters": {"conf": 0.5}}},
)
print(res.images)       # [bytes, bytes]  annotated JPEGs
print(res.detections)   # already decoded to a list of dicts (json codec)
per_image = {}
for d in res.detections:
    per_image.setdefault(d["image_index"], []).append(d)

# 2) tracking (stateful — ids keep accumulating across calls)
b.run(config={"yolo": {"command": "reset"}})                # clean slate
for frame in frames:
    r = b.run(data={"images": [frame]},
              config={"yolo": {"command": "track",
                               "parameters": {"tracker": "botsort.yaml"}}})
    for d in r.detections:
        print(d["track_id"], d["bbox"], d["class_name"])
```

Because `images` is declared `identity` and `detections` is declared `json`,
`boxes_client` hand you `bytes` and a ready Python list respectively — no
manual decode.

## Test

```bash
# in-process, no GPU / no weights (fake Ultralytics model)
python test/test_yolo_fake.py

# live, against a running box
python test/test_yolo.py
BOX_HOST=localhost:8061 python test/test_yolo.py
```
