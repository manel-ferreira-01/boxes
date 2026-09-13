# boxes

A fleet of independent, Dockerized AI inference services — "boxes" — speaking
one shared gRPC envelope, plus a thin, box-agnostic Python client that drives
any of them.

One box = one container = one gRPC service:

```protobuf
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

Every box listens on port **8061** (AI4EU spec; overridable with the `PORT`
env var), dispatches work from a `config` section namespaced under its own key
(`{"clip": {...}}`, `{"lang_sam": {...}}`, …), and answers with a namespaced
status plus `data` payloads. There is no per-box SDK and no central
orchestrator — boxes stay independent, addressable, composable units, and the
*client* does the calling.

## Calling a box

```python
from boxes_client import Box
import pathlib

b = Box("localhost:8061")                      # any box, by IP:port
res = b.run(data   ={"images": [pathlib.Path("dog.jpg")]},
            config ={"lang_sam": {"command": "segment",
                                  "parameters": {"box_threshold": 0.3,
                                                 "text_threshold": 0.25},
                                  "text_prompt": ["a dog"]}})
print(res)          # decoded fields + parsed config + status
print(res.results)  # decoded payload (the box declared "encoding": "zstd_pickle")
```

That's the entire end-user surface — zero box knowledge in the call.
Details: [boxes_client/README.md](boxes_client/README.md).

## Quick start

Run the five-box fleet (host ports 9061–9065) and tour it:

```bash
cd fleet
docker compose up -d
python hello.py             # the minimal tour: one generic call per box
python supervisor_demo.py   # guided demo with decoded output per box
```

Or run a single box yourself (full instructions in
[docs/Quick_Start_Guide.md](docs/Quick_Start_Guide.md)):

```bash
cd images/lang_segm
docker build --tag my_lang_segm -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -e PORT=8061 --ipc=host my_lang_segm
```

## Boxes in this repo

| Box | Type | What it does |
|-----|------|--------------|
| clip | GPU | CLIP image/text embeddings |
| tapnext_tracker | GPU | Point tracking with TAPNext (stateful) |
| lang_segm | GPU | Text-guided segmentation (LangSAM) |
| textEmbedding | GPU / CPU | Sentence-BERT text embeddings |
| vggt | GPU | 3D reconstruction from image sequences |
| yologpt | GPU | YOLOv11 detection & tracking |
| opencv_box | GPU / CPU | Optical flow, feature matching, similarity checks |
| folder_wd | CPU | File-watcher: watches a directory, drives boxes, saves outputs |
| gradio_display | CPU | Gradio UI: upload/images, drive boxes, display results |

**The per-box README is the authoritative source for that box's request shape**
(config keys, fields, status, how to decode `results`):
[`images/<name>/README.md`](images/).

## Key contracts

- **Shared envelope** — [`protos/pipeline.proto`](protos/) is the one
  interface, vendored into every box; `aux.py` is the `wrap_value` /
  `unwrap_value` helper.
- **Config dispatch** — the request's `config_json` carries a section
  namespaced under the box's key; `{"command": "reset"}` is accepted by every
  standard box.
- **Declared payload encoding** — boxes declare `"encoding"` (a codec name, or
  a `{field: codec}` map) in their *response* config; the client decodes with
  the named codec (`identity` / `json` / `torch` / `numpy` / `zstd_pickle`)
  and defaults to raw `bytes`. Design + rationale:
  [docs/CODECS.md](docs/CODECS.md).

## Documentation

Start in [docs/index.md](docs/index.md):

- [Architecture overview](docs/Architecture_Overview.md) — what a box is,
  conventions, GPU memory lifecycle
- [Quick start guide](docs/Quick_Start_Guide.md) — run a box, call it, build
  your own, test it
- [gRPC services reference](docs/gRPC_Services_Reference.md) — the envelope
  contract: `config_json` shape per box, `data` fields, `results` encoding
- [Docker image template guide](docs/Docker_Image_Template_Guide.md) — the
  Dockerfile templates used by the boxes in this repo
- [CODECS](docs/CODECS.md) — self-describing payload decoding

## Tests

- **Client** — `boxes_client/tests/`: `fake_box_smoke.py` (in-process fake
  boxes, no GPU) and `codec_smoke.py` (the declared-encoding contract), plus
  `live_tapnext.py` against a real box.
- **Boxes** — each box ships `images/<name>/test/test_<name>.py`, pointable at
  a running box:
  `python images/lang_segm/test/test_lang_sam.py`
