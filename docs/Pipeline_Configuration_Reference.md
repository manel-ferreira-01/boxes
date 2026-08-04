# Maestro Pipeline Configuration Reference

## Overview

Maestro orchestrates gRPC services into data processing pipelines using **YAML configuration files**.

Configuration format: multiple YAML documents (separated by `---`) defining resources:
- `kind: pipeline` - Top-level pipeline definition
- `kind: stage` - Individual service endpoints  
- `kind: link` - Data flow connections

## Configuration File Structure

### Example Pipeline

```yaml
# Define the pipeline
kind: pipeline
spec:
  name: FullGradioPipeline
---
# Stage 1: Acquire input
kind: stage
spec:
  name: gradio_acquire
  method: acquire_yolo_detect
  address: interface:8061
  pipeline: FullGradioPipeline
---
# Stage 2: Process with YOLO
kind: stage
spec:
  name: yolo_detect
  method: DetectSequence
  address: yologrpc:8061
  pipeline: FullGradioPipeline
---
# Stage 3: Display results
kind: stage
spec:
  name: gradio_display
  method: display_yolo_detect
  address: interface:8061
  pipeline: FullGradioPipeline
---
# Link: data flow from acquire → yolo
kind: link
spec:
  name: acquire-to-yolo
  source_stage: gradio_acquire
  target_stage: yolo_detect
  pipeline: FullGradioPipeline
---
# Link: data flow from yolo → display
kind: link
spec:
  name: yolo-to-display
  source_stage: yolo_detect
  target_stage: gradio_display
  pipeline: FullGradioPipeline
```

## Resource Types

### Pipeline Resource

Defines a named pipeline (workflow).

```yaml
kind: pipeline
spec:
  name: <unique-pipeline-name>
```

**Fields:**
- `name` (required): Unique identifier for the pipeline

---

### Stage Resource

Represents a single gRPC service endpoint in the pipeline.

```yaml
kind: stage
spec:
  name: <stage-name>
  method: <grpc-method>
  address: <host>:<port>
  pipeline: <pipeline-name>
```

**Fields:**
- `name` (required): Unique stage identifier within the pipeline
- `method` (optional): gRPC method name to call. Defaults to single available method.
- `address` (required): Hostname and port of service (`service_name:8061`)
- `pipeline` (required): Name of pipeline this stage belongs to

---

### Link Resource

Defines data flow between two stages.

```yaml
kind: link
spec:
  name: <link-name>
  source_stage: <source-stage-name>
  target_stage: <target-stage-name>
  pipeline: <pipeline-name>
```

**Fields:**
- `name` (required): Unique identifier for this link
- `source_stage` (required): Name of stage that produces data
- `target_stage` (required): Name of stage that receives data
- `pipeline` (required): Pipeline this link belongs to

---

## Real-World Examples

### Example 1: Simple Detection Pipeline

**Use case:** Acquire → Detect → Display (single pass)

```yaml
kind: pipeline
spec:
  name: SimpleDetect
---
kind: stage
spec:
  name: acquire
  method: acquire_yolo_detect
  address: interface:8061
  pipeline: SimpleDetect
---
kind: stage
spec:
  name: detect
  method: DetectSequence
  address: yologrpc:8061
  pipeline: SimpleDetect
---
kind: stage
spec:
  name: display
  method: display_yolo_detect
  address: interface:8061
  pipeline: SimpleDetect
---
kind: link
spec:
  name: a-to-detect
  source_stage: acquire
  target_stage: detect
  pipeline: SimpleDetect
---
kind: link
spec:
  name: detect-to-display
  source_stage: detect
  target_stage: display
  pipeline: SimpleDetect
```

---

### Example 2: Tracking with Flow Control

**Use case:** Track objects across video frames, only process when content changes

```yaml
kind: pipeline
spec:
  name: VideoTracker
---
# Acquire frame from file watcher
kind: stage
spec:
  name: acquire_frame
  method: acquire
  address: folder_wdgrpc:8061
  pipeline: VideoTracker
---
# Check if frame changed significantly
kind: stage
spec:
  name: similarity_check
  method: similarity_check
  address: opencv_service:8061
  pipeline: VideoTracker
---
# Run YOLO only on changed frames
kind: stage
spec:
  name: track
  method: TrackSequence
  address: yologrpc:8061
  pipeline: VideoTracker
---
kind: link
spec:
  name: frame-to-check
  source_stage: acquire_frame
  target_stage: similarity_check
  pipeline: VideoTracker
---
kind: link
spec:
  name: check-to-track
  source_stage: similarity_check
  target_stage: track
  pipeline: VideoTracker
```

**Logic:** OpenCV similarity check returns "changed" flag. YOLO only runs when `changed=True`.

---

## Stage Method Reference

### By Service Type

#### Gradio Display Service (`interface`)
```protobuf
rpc acquire_yolo_detect(Envelope) returns (Envelope)
rpc acquire_yolo_track(Envelope)  returns (Envelope)
rpc acquire_vggt(Envelope)        returns (Envelope)

rpc display_yolo_detect(Envelope) returns (Envelope)
rpc display_yolo_track(Envelope)  returns (Envelope)
rpc display_vggt(Envelope)        returns (Envelope)
```

#### YOLO Service (`yologrpc`)
```protobuf
rpc DetectSequence(Envelope) returns (Envelope)
rpc TrackSequence(Envelope)  returns (Envelope)
```

#### OpenCV Service (`opencv_service`)
```protobuf
rpc Process(Envelope)           returns (Envelope)
rpc similarity_check(Envelope)  returns (Envelope)
```

---

## Common Patterns

### Pattern: Batch Processing

YOLO services handle batches automatically - send multiple images in one envelope.

### Pattern: Stream Processing with Countdown

For video processing, use `stream` flag set to frame count countdown:

```python
Envelope(
    config_json=json.dumps({
        "stream": total_frames - current_frame  # countdown to end
    }),
    data={"images": wrap_value([current_frame_bytes])}
)
```

The stream flag decrements; when it reaches 0, reset tracker/state.

---

## Validation Rules

### Required Fields Per Resource

| Resource | Required Fields |
|----------|-----------------|
| pipeline | `name` |
| stage | `name`, `address`, `pipeline` |
| link | `name`, `source_stage`, `target_stage`, `pipeline` |

### Name Constraints
- All names must be unique within a pipeline
- No spaces or special characters (use underscores)
- Case-sensitive

---

## Deployment Example: Docker Compose

```yaml
version: '3.8'
services:
  interface:
    image: sipgisr/displaygrpc
    ports:
      - "7860:7860"
  
  yologrpc:
    image: sipgisr/yologrpc
  
  vggtgrpc:
    image: sipgisr/vggtgrpc
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  
  maestro:
    image: sipgisr/maestro:v1-latest
    volumes:
      - type: bind
        source: ./config.yaml   # Your pipeline config
        target: /config.yaml
```

**Start:** `docker-compose up -d`

---

## Debugging Tips

### Check Service Availability
```bash
python3 -c "
import grpc
from protos import pipeline_pb2, pipeline_pb2_grpc
channel = grpc.insecure_channel('localhost:8061')
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)
print(stub._method_handlers.keys())
"
```

### Validate YAML Syntax
```bash
python3 -c "import yaml; yaml.safe_load_all(open('config.yaml'))"
```

### View Maestro Logs
```bash
docker logs <maestro_container>
```

---

## Complete Example: Full Pipeline

See `/home/manuelf/boxes/pipelines/gradio+vggt+yolo/config.yaml` for the most complex example in your codebase.
