# TAPNext Tracker gRPC Service

A gRPC service for TAPNext point tracking with streaming support, following the vggt image pattern.

## Features

- Frame-by-frame tracking with state preservation
- Grid-based query point detection on first frame
- Server-side track accumulation until reset
- **Multi-session (multi-tenant)**: one box serves many independent users at
  once — each `session_id` holds its own state, its own reset, and its own
  accumulation (see [Sessions](#sessions-sharing-one-box-with-many-users))
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

Build without checkpoint (service checks for checkpoint at runtime):

```bash
docker build --tag sipgisr/tapnexttracker \
    --build-arg SERVICE_NAME=tapnext \
    -f docker/Dockerfile .
```

Or mount the checkpoint at runtime:

```bash
docker run -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz \
    sipgisr/tapnexttracker
```

## Service Usage

### gRPC Protocol

The service uses the shared `PipelineService` interface with `Envelope` messages.

#### Reset Tracking State (one-time initialization)

Reset clears previous tracking state before starting a new sequence. Reset is
**scoped to a session**: without a `session_id` it clears the shared `default`
session; pass one and only that user's state is touched (others keep running):

```python
import grpc
import pipeline_pb2 as proto
from aux import wrap_value, unwrap_value

# Initialize gRPC client
channel = grpc.insecure_channel('localhost:8061')
stub = proto.PipelineServiceStub(channel)

request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {"command": "reset"}          # + "session_id": "alice-2025" to scope
    })
)
response = stub.Process(request)
```

#### Tracking Request

Track points in a single frame. The service automatically tracks across sequential frames:

```python
# Prepare frame data
with open('frame.jpg', 'rb') as f:
    frame_bytes = f.read()

request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {
            "command": "track",
            "parameters": {"grid_size": 32}
        }
    }),
    data={"images": [wrap_value(frame_bytes)]}
)

response = stub.Process(request)
tracks = unwrap_value(response.data["tracks"])
visibles = unwrap_value(response.data["visibles"])
```

Each frame you send continues from the previous tracking state. No reset flag needed between frames.

### Sessions (sharing one box with many users)

The box is **multi-tenancy-ready**: every request can carry a `session_id`
(inside the `tapnext` config section). All requests with the same `session_id`
see the same tracking state; different ids see only their own — state,
accumulation, and reset are all scoped per session. No login or registry is
involved: the id is an opaque **capability string** — knowing it is enough to
run against that session, and it is the only way to name one.

```python
from boxes_client import Box
import pathlib

b = Box("localhost:9063")

# Student "alice" tracks — her session is created on first use
res = b.run(
    data   = {"images": [pathlib.Path("frame1.jpg")]},
    config = {"tapnext": {"command": "track",
                          "parameters": {"grid_size": 32},
                          "session_id": "alice-2025"}},
)

# Student "bob" on the SAME box, same GPU: completely independent
b.run(data={"images": [pathlib.Path("frame1.jpg")]},
      config={"tapnext": {"command": "track",
                          "parameters": {"grid_size": 32},
                          "session_id": "bob-2025"}})

# Reset is scoped: only alice's session restarts; bob's keeps going
b.run(config={"tapnext": {"command": "reset", "session_id": "alice-2025"}})

# Operator view: which sessions are alive (sid + frames + idle time, no state)
b.run(config={"tapnext": {"command": "list"}})
```

| Behaviour | Single session (no `session_id`) | Named sessions |
|---|---|---|
| State/accumulation | one shared `default` session | one per `session_id` |
| `reset` | clears the `default` session | clears only that session |
| Back-compat | old clients keep working unchanged | new key, ignored by old boxes |

Notes:
- Omitting `session_id` (or sending `null`) runs in the shared **`default`**
  session — the pre-multi-session behaviour, so existing callers are untouched.
- A session stays alive until it is reset, reaped by `TAPNEXT_SESSION_TTL` (see
  env vars), or the box is restarted. By default sessions are kept forever,
  which is what you want in a classroom.
- The `session_id` is a capability: anyone who can reach the box port and knows
  an id can continue or reset that session (fine on a trusted LAN — do not
  publish unknown ids publicly if you don't trust the audience). `list` lets
  someone who reaches the box enumerate the active ids.
- The response config echoes which session answered (`"session": "<sid>"`).

#### Sessions in raw gRPC (without the client)

```python
request = proto.Envelope(
    config_json=json.dumps({
        "tapnext": {
            "command": "track",
            "parameters": {"grid_size": 32},
            "session_id": "alice-2025"
        }
    }),
    data={"images": [wrap_value(frame_bytes)]}
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `command` | string | "track" | "track" for inference, "reset" to clear *this session's* tracking state, "list" to see active sessions (operator) |
| `session_id` | string | `"default"` | Opaque session tag; gives the user a private state. Share an id = share a session |
| `grid_size` | int | 32 | Number of grid points per dimension (grid_size × grid_size total) |

### Response Format

Response includes all tracked points and the Tomasi-Kanade observation matrix:

```json
{
  "tapnext": {
    "status": "done",
    "session": "alice-2025",
    "runtime": 0.45,
    "frames_processed": 1,
    "num_points": 1024
  }
}
```

`"session"` echoes the id that answered (or `"default"`), so a client can
confirm which session it just talked to.

A `list` request returns a config-only envelope (no `data`):

```json
{
  "tapnext": {
    "status": "done",
    "action": "list",
    "sessions": [
      {"session": "alice-2025", "frames_processed": 42, "num_tracks": 1024, "idle_seconds": 3.2},
      {"session": "bob-2025",   "frames_processed": 7,  "num_tracks": 1024, "idle_seconds": 190.5}
    ]
  }
}
```

Response data contains:
- `tracks`: Float32 array of shape (frame_count, num_points, 2)
- `visibles`: Float32 array of track visibility logits  
- `observation_matrix`: Tomasi-Kanade P matrix (2×frames × num_points) for factorization

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
| `TAPNEXT_SESSION_TTL` | 0 (keep forever) | Seconds a session may sit idle before it (and its per-session GPU state) is reaped. 0 disables reaping — the right choice for a classroom where students' work must persist. Raise it (e.g. 1800) on a shared GPU to reclaim VRAM from abandoned sessions. |

## Performance Notes

- First inference call loads the model (~740MB checkpoint)
- Model stays on GPU after loading unless idle for 120 seconds
- Grid detection generates (grid_size × grid_size) query points per frame
- Track state persists across sequential frames for a session until that session
  is reset or reaped
- With multiple sessions the shared model is loaded once; each session adds a
  small per-session state cost. Reap idle sessions with `TAPNEXT_SESSION_TTL`
  if many sessions accumulate on one GPU.

## Testing

Two test entry points:

```bash
# In-process session-isolation suite — NO GPU, NO tapnet wheel, NO Docker.
# Injects a deterministic stub model and exercises Process() directly (plus a
# real gRPC round-trip). This is the fastest way to prove multi-session
# isolation/regression.
cd images/tapnext_tracker/test
python test_tapnext_sessions.py

# Live smoke test against a running box (needs the real image + a GPU):
docker build --tag my_tapnext -f docker/Dockerfile .
docker run --rm --gpus all -p 8061:8061 -v /path/to/ckpt:/workspace/bootstapnext_ckpt.npz my_tapnext &
cd images/tapnext_tracker/test
python test_tapnext.py          # sequential tracking over gRPC
```

## Troubleshooting

### Checkpoint Not Found

Ensure checkpoint is at `/workspace/bootstapnext_ckpt.npz`:

```bash
docker run -v /path/to/checkpoint:/workspace/bootstapnext_ckpt.npz tapnexttracker
```

### CUDA Out of Memory

Reduce batch size by processing frames individually or use smaller grid_size.
