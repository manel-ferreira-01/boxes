# Pipeline Architecture Overview

## What is This System?

This is a **distributed AI processing pipeline system** that connects Dockerized gRPC services into workflow pipelines orchestrated by Maestro.

### Core Concepts

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Image 1   │────>│   Image 2   │────>│   Image 3   │────>│   Image N   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     │                   │                   │                   │
     ▼                   ▼                   ▼                   ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          Maestro Orchestrator                        │
│                    (Coordinates data flow between stages)             │
└───────────────────────────────────────────────────────────────────────┘
```

## System Components

### 1. Docker Images (`/images/`)

Each directory in `/home/manuelf/boxes/images/` contains a self-contained AI service:

```
images/
├── opencv_box/        # Feature matching, optical flow
├── vggt/              # 3D reconstruction from images
├── yologpt/           # YOLO detection & tracking
├── cotracker/         # Video motion tracking
├── gradio_display/    # Web UI for user interaction
└── textEmbedding/     # Text embeddings (Sentence-BERT)
```

**Each image contains:**
- Dockerfile for building the container
- `protos/*.proto` - gRPC service definitions
- `src/*_service.py` - Business logic implementation
- `requirements.txt` - Python dependencies

### 2. gRPC Protobuf Protocol

All services communicate using **standardized message envelopes**:

```protobuf
message Envelope {
  string config_json = 1;                // JSON metadata/config
  map<string, Value> data = 2;           // Named data fields
}

message Value {
  oneof kind { 
    bytes b       = 1;
    string s      = 2;
    float f       = 5;
    BytesList bb  = 6;
    StringList ss = 7;
    FloatList ff  = 10;
  }
}
```

This envelope pattern allows:
- Flexible data types (images, tensors, metadata)
- Version-independent communication
- Easy inspection and debugging

### 3. Maestro Orchestrator

**Location:** `/home/manuelf/maestro`

Maestro reads pipeline configuration files and manages:

#### Pipeline
Top-level workflow definition:
```yaml
kind: pipeline
spec:
  name: YoloPipeline
```

#### Stage
A single gRPC service endpoint in the pipeline:
```yaml
kind: stage
spec:
  name: yolo_detect
  method: DetectSequence
  address: yologrpc:8061
  pipeline: YoloPipeline
```

#### Link
Data flow connections between stages:
```yaml
kind: link
spec:
  name: data-flow-1
  source_stage: gradio_acquire
  target_stage: yolo_detect
  pipeline: YoloPipeline
```

## Data Flow Patterns

### Pattern 1: Acquisition → Processing → Display

```
Gradio UI (acquire) ──▶ YOLO (process) ──▶ Gradio UI (display)
```

Use case: User uploads images, gets back annotated results.

### Pattern 2: Video Stream Processing

```
File Watcher (acquire) ──▶ OpenCV (frame analysis) ──▶ File output
                             ↓
                        Video analysis
```

Use case: Process video files dropped in a folder, save processed frames.

### Pattern 3: Multi-Stage Analysis

```
Gradio (input) ──▶ VGGT (3D recon) ──▶ Gradio (display)
                              ↓
                         OpenCV (verification)
```

Use case: Complex multi-algorithm workflows.

## Service Types

### 1. **Acquire Services**
Initiate data flow, capture input:
- `acquire` - Get next data packet
- `acquire_vggt`, `acquire_yolo_detect`, etc.
- Often implemented in Gradio layer or file watchers

### 2. **Process Services**
Core AI/ML inference:
- `Process` - Generic processing
- `DetectSequence`, `TrackSequence` - YOLO-specific
- `Forward` - Embeddings, similarity
- `similarity_check` - Image comparison

### 3. **Display Services**
Output/results handling:
- `display` - General output
- `display_yolo_detect`, `display_vggt`, etc.
- Update UI or save results to files

## Architecture Principles

### Standardization
- **Port 8061** - All services listen on same port (AI4EU spec)
- **Envelope messages** - Consistent message format across all services
- **Docker containers** - Isolated, reproducible deployments

### Modularity
- Each service is independent
- Services can be reused in different pipelines
- Easy to swap algorithms without changing pipeline config

### Scalability
- Services can run on different machines
- Maestro handles service discovery via config
- Support for GPU acceleration where needed

## Workflow Example: YOLO Detection Pipeline

```yaml
# Pipeline definition (config.yaml)
kind: pipeline
spec:
  name: YoloPipeline
---
kind: stage
spec:
  name: acquire
  method: acquire_yolo_detect
  address: interface:8061
  pipeline: YoloPipeline
---
kind: stage
spec:
  name: detect
  method: DetectSequence
  address: yologrpc:8061
  pipeline: YoloPipeline
---
kind: stage
spec:
  name: display
  method: display_yolo_detect
  address: interface:8061
  pipeline: YoloPipeline
---
kind: link
spec:
  name: acquire-to-detect
  source_stage: acquire
  target_stage: detect
  pipeline: YoloPipeline
---
kind: link
spec:
  name: detect-to-display
  source_stage: detect
  target_stage: display
  pipeline: YoloPipeline
```

### Execution Flow:

1. **Start Maestro** with config file
2. **User interacts** with Gradio UI at `interface:7860`
3. **Gradio** writes request to temp file
4. **acquire stage** reads from temp file, sends images to yologrpc
5. **detect stage** runs YOLO on images, returns annotated frames
6. **display stage** sends results back to Gradio UI

## Benefits of This Architecture

| Benefit | Explanation |
|---------|-------------|
| **Framework-agnostic** | Services can use any AI framework (PyTorch, TensorFlow) |
| **Language-independent** | gRPC works across languages (Python, Java, C++) |
| **Restartable** | Services can be restarted independently |
| **Scalable** | Multiple instances of same service for load balancing |
| **Debuggable** | Standardized messages are easy to inspect/logs |
| **Testable** | Services work standalone or in pipelines |

## Summary

This system provides:
1. ✅ **Reusable AI services** as Docker containers
2. ✅ **Standardized communication** via gRPC + protobuf
3. ✅ **Flexible orchestration** via Maestro YAML configs
4. ✅ **Production-ready** with GPU support, error handling, idle management
