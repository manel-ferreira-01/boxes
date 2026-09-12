# boxes-client

A thin Python client for calling deployed AI **"boxes"** by `IP:port`.

A *box* is one of the gRPC services in [`/images/`](../images/) built to the
shared

```protobuf
service PipelineService {
  rpc Process( Envelope ) returns ( Envelope );
}
```

interface. Point this client at **any one box** and send an `Envelope`:

```python
from boxes_client import Box

b = Box("localhost:8061")                 # a local box
# b = Box("10.0.0.5:8061")                # ...or a remote one

res = b.run(data={"images": ["frame.jpg"]},
            config={"my_box": {"command": "do_thing", "parameters": {}}})
print(res.fields)     # decoded payload, whatever the box returned
print(res.config)     # parsed config_json
```

Box-specific one-liners live in an *optional* convenience layer, e.g. the
tapnext tracker:

```python
from boxes_client import trace
res = trace(b, images=["frame.jpg"], grid_size=30)
```

No registry, no central server — the client connects directly to the box, so
local and remote boxes are the same call. Boxes stay independent, addressable,
composable units (the *client* does the calling, preserving the distributed
nature of the fleet).

## Install

```bash
pip install -e boxes_client
# optional, to decode tapnext/vggt tensor payloads to numpy:
pip install -e "boxes_client[torch]"
```

Without `torch`, tensor fields are returned as raw `bytes` (still usable —
`torch.load(BytesIO(res.tracks), weights_only=False)`).

## Core API (box-agnostic) + optional conveniences

`Box` is deliberately box-agnostic: it builds and sends an `Envelope` and reads
a `Result` back, and knows **no box, field, or model**. Per-box sugar lives in a
separate convenience layer, so adding one never touches the core.

### 1. Generic — `Box.run(data, config, method, reset_first)`
The workhorse. No assumption about field names or payload types:

```python
b.run(
    data    = {"images":    [img1, img2]},           # any field names; any types
    config  = {"my_box":    {"command": "do_thing", "parameters": {...}}},
    method  = "Process",                              # default; e.g. "similarity_check"
)
```

`data` is a dict of `field_name -> value`. Values are coerced per the rules
below and wrapped into the shared `Value` oneof via the vendored `aux.wrap_value`.

### 2. Convenience — `trace(box, images, grid_size=None, reset_first=True)`
An **optional** one-liner for the *tapnext* box, living in
`boxes_client.conveniences` (not the core). It reads image files (or accepts
pre-encoded bytes) and just calls `Box.run()` with the right shape:

```python
b = Box("localhost:8061")
res = trace(b, images=["f1.jpg", "f2.jpg", "f3.jpg"], grid_size=30)
np.save("tracks.npy", res.tracks.numpy())
```

The core `Box` knows nothing about tapnext — `trace` *is* the only tapnext
knowledge, and it's safe to delete without touching the generic client. Add a
sibling convenience (`segment`, `embed`, `detect`, …) for other boxes the same
way; never put a box name in `box.py`.

### `Box.reset(config_key=None)`
Sends `{config_key: {"command": "reset"}}` on `Process`. `config_key` is the
box's section name (or the one from the constructor). Stateful boxes clear
state; stateless boxes typically ignore it. Pass the box name explicitly, or
construct with `Box(host, config_key="tapnext")`.

### `Box.info()`
Asks the box (via gRPC reflection) whether it serves `pipeline.PipelineService`.
Useful to check box reachability and shape before committing to a call.

## Value coercion (for `Box.run` / convenience `data` values)

| You pass | What's sent |
|---|---|
| `bytes` / `bytearray` / `memoryview` | `b` (single bytes) or `bb` (list) — for pre-encoded binary payloads (images, tensors, etc.) |
| `pathlib.Path` | file bytes (read locally, sent as `b`) |
| `str` | **literal string** (NOT a file path) — sent as `s` |
| `int` | coerced to `float` (proto `f` is a float) |
| list of the above (homogeneous) | corresponding list: `BytesList` / `StringList` / `FloatList` |

That's why a *file* has to be passed as `pathlib.Path` (or a `bytes` you loaded
yourself) — the client serializes it and ships it. A *literal string* is just
`str` — no confusion.

```python
import pathlib
# file:
b.run(data={"images": [pathlib.Path("frame.jpg")]}, config={...})

# preencoded:
b.run(data={"images": [open("frame.jpg","rb").read()]}, config={...})

# literal text (e.g. a hypothetical text-only envelope box):
b.run(data={"sentences": ["hello", "world"]}, config={...})

# list of floats:
b.run(data={"vals": [1, 2, 3]}, config={...})       # -> FloatList
```

## Result object

`Result` wraps the raw response `Envelope`:

- `fields` — dict of `field_name -> best-effort decoded value`
- `.tracks`, `.visibles`, … — direct field access via `__getattr__`
- `config` — parsed `config_json`
- `raw` — the undecoded `Envelope` proto
- `as_dict()` — JSON-friendly version (numpy arrays -> `tolist`)

Decoding order (`[src/boxes_client/decode_util.py](src/boxes_client/decode_util.py)`):
**JSON → torch (if installed) → numpy → raw bytes.** Nothing raises.

## What's supported

| Box | In scope | Notes |
|-----|----------|-------|
| tapnext (`Process`) | ✅ | v1 target; `trace(box, ...)` convenience over `Box.run` |
| vggt, yolo, opencv_box, lang_segm, clip (`Process`) | ✅ envelope shape | call via `Box.run(...)` with the box-specific `config`; extras like yolo `DetectSequence` / opencv `similarity_check` need the box's own proto for `method=` |
| cotracker, textEmbedding (`Forward`) | ⏸ pending | use `Box.run` after they're migrated to the shared envelope (client needs no changes) |

### Method dispatch caveat

`Box.run(..., method=...)` resolves to a method on the **client-stub**, which is
built from the shared `pipeline.proto`. Today the shared proto defines only
`Process`, so `method=` is currently useful only for `Process`. Boxes that add
extra `PipelineService` RPCs (e.g. opencv's `similarity_check`, yolo's
`DetectSequence`/`TrackSequence`/`AllProcessing`) will need either (a) to also
be migrated to a *single* `Process` with a `command` field (the tapnext pattern),
or (b) to have their own proto vendored into the client. This is the only
remaining "per-box" knowledge in the client — everything else (field names,
payload types, config shape) is fully generic.

## Run the tests

```bash
# in-process fake box (no GPU, no real box needed)
python boxes_client/tests/fake_box_smoke.py

# real tapnext box at BOX_HOST:PORT
BOX_HOST=localhost:8061 python boxes_client/tests/live_tapnext.py
```

## Notes / design

- **Client-side only.** The box stays a box: independent, restartable,
  composable. Distribution is preserved because the *client* dials the box
  directly, so "local box" and "remote box" are the same call.
- **Auto-protocol.** `Box.info()` uses gRPC reflection to confirm the box
  serves `pipeline.PipelineService`. If the box does not serve reflection,
  calls still work — `info()` just reports `reflection: False`.
- **Best-effort decoding.** JSON → torch → numpy → raw bytes (see above).
  Nothing raises; you always get a `Result`.
- **Forward boxes deferred.** cotracker / textEmbedding use bespoke
  `Forward` messages; they will work through this client with no changes once
  migrated to the shared envelope (clip is already migrated).
