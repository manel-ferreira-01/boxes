# How gRPC Services Work in This System

## Overview

Each AI service runs as a Docker container exposing a **gRPC server** on port 8061. Services implement the `PipelineService` interface with custom methods.

## Service Interface Standard

### Proto Definition

```protobuf
syntax = "proto3";

package pipeline;

message Envelope {
  string config_json = 1;                
  map<string, Value> data = 2;      
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

service PipelineService {
  rpc Process(Envelope) returns (Envelope);
  // Additional custom methods defined per service
}
```

### Data Types

The `Value` type supports:
- **Scalars:** bytes, string, float
- **Lists:** BytesList, StringList, FloatList (repeated)

## Service Implementation Pattern

### Basic Structure

```python
import grpc
from concurrent import futures

class MyService(<service>_pb2_grpc.<Service>Servicer):
    def __init__(self):
        # Load models here (once at startup)
        self.model = load_model()
    
    def <MethodName>(self, request, context):
        # 1. Parse Envelope
        config = json.loads(request.config_json)
        images = unwrap_value(request.data.get("images", []))
        
        # 2. Process data
        results = self.model.infer(images)
        
        # 3. Return response
        return <service>_pb2.Envelope(
            config_json=json.dumps({"status": "success"}),
            data={"results": wrap_value(results)}
        )

# Server setup
server = grpc.server(futures.ThreadPoolExecutor())
<service>_pb2_grpc.add_<Service>Servicer_to_server(MyService(), server)
server.add_insecure_port('[::]:8061')
server.start()
server.wait_for_termination()
```

## Service Method Categories

### 1. Common Methods (PipelineService)

| Method | Purpose |
|--------|---------|
| `Process(Envelope)` | Generic processing - accepts Envelope, returns Envelope |

### 2. Custom Methods by Service

#### YOLO (`yologpt`)
```protobuf
service PipelineService {
  rpc DetectSequence(Envelope) returns (Envelope);   # Run detection on batch
  rpc TrackSequence(Envelope) returns (Envelope);    # Track objects across frames
}
```

Usage:
```python
# Detection
response = stub.DetectSequence(
    Envelope(
        config_json=json.dumps({"threshold": 0.5}),
        data={"images": wrap_value([img1_bytes, img2_bytes])}
    )
)

# Tracking
response = stub.TrackSequence(
    Envelope(
        config_json=json.dumps({"stream": 0}),  # stream flag
        data={"images": wrap_value([frame1, frame2, ...])}
    )
)
```

#### OpenCV (`opencv_box`)
```protobuf
service PipelineService {
  rpc Process(Envelope) returns (Envelope);           # Generic processing
  rpc similarity_check(Envelope) returns (Envelope);  # Frame comparison
}
```

Usage:
```python
# Feature matching between two images
response = stub.Process(
    Envelope(
        config_json=json.dumps({"feature_extractor": "SuperPoint"}),
        data={"images": wrap_value([img1_bytes, img2_bytes])}
    )
)
```

#### CoTracker (`cotracker`)
```protobuf
service CoTrackerService {
  rpc Forward(CoTrackerRequest) returns (CoTrackerResponse);
}

message CoTrackerRequest {
  bytes video = 1;
  int32 grid_size = 4;
}
```

## Helper Functions: wrap_value / unwrap_value

### Purpose
Convert Python objects to/from protobuf `Value` type.

### Implementation

```python
def wrap_value(obj):
    """Wrap Python object into pipeline.Value"""
    if isinstance(obj, bytes):
        return Value(b=obj)
    elif isinstance(obj, str):
        return Value(s=obj)
    elif isinstance(obj, float):
        return Value(f=obj)
    elif isinstance(obj, list) and all(isinstance(v, bytes) for v in obj):
        return Value(bb=BytesList(values=obj))
    # ... more types
```

### Usage Examples

```python
# Single image (bytes)
data={"images": wrap_value(image_bytes)}

# Multiple images (list of bytes)
data={"images": wrap_value([img1, img2, img3])}

# Dictionary metadata
data={"metadata": wrap_value({"frame": 5, "timestamp": "..."})}

# List of floats
data={"distances": wrap_value([0.1, 0.5, 0.9])}
```

### Unwrapping Data

```python
# Get images from envelope
images = unwrap_value(request.data.get("images", []))

# Get metadata
config = json.loads(request.config_json)
threshold = config.get("threshold", 0.5)
```

## Service Lifecycle

### 1. startup (once at container start)
- Load models into memory
- Initialize CUDA context (if GPU)
- Set up file watchers or database connections

### 2. Request Handling (repeated)
```python
def Process(self, request, context):
    # Parse input
    images = unwrap_value(request.data.get("images"))
    
    # Run inference
    results = self.model(images)
    
    # Format output
    return Envelope(data={"predictions": wrap_value(results)})
```

### 3. shutdown (on container stop)
- Release GPU memory
- Save state if needed
- Cleanup resources

## Error Handling Pattern

```python
def Process(self, request, context):
    try:
        images = unwrap_value(request.data.get("images", []))
        if not images:
            raise ValueError("No images provided")
        
        results = self.model.infer(images)
        
        return Envelope(data={"results": wrap_value(results)})
    
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        # Return empty envelope to signal failure
        return Envelope()
```

## Performance Considerations

### Maximally Flexible Messages
Services set large message limits:

```python
server = grpc.server(
    futures.ThreadPoolExecutor(),
    options=[
        ('grpc.max_send_message_length', -1),
        ('grpc.max_receive_message_length', -1),
        ('grpc.max_message_length', -1)
    ]
)
```

### GPU Memory Management

**Idle timeout pattern:**
```python
class MyService(<service>_pb2_grpc.<Service>Servicer):
    def __init__(self):
        self._model = load_model_to_gpu()
        self._last_request_time = time.time()
        
        # Watchdog to move back to CPU when idle
        threading.Thread(target=self._watchdog_loop, daemon=True).start()
    
    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            idle = time.time() - self._last_request_time
            if idle > 60 and self._device.startswith("cuda"):
                self._model.cpu()  # Free GPU memory
                torch.cuda.empty_cache()
    
    def Process(self, request, context):
        with self._lock:
            self._last_request_time = time.time()
            
            # Move to GPU if needed
            if self._device == "cpu" and torch.cuda.is_available():
                self._model.cuda()
        
        return self._run_inference(request)
```

## Testing Services

### Standalone (outside pipeline)

```python
import grpc
from protos import pipeline_pb2, pipeline_pb2_grpc

channel = grpc.insecure_channel('localhost:8061')
stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

response = stub.Process(
    pipeline_pb2.Envelope(
        config_json='{"param": "value"}',
        data={"images": wrap_value([test_image_bytes])}
    )
)

results = unwrap_value(response.data.get("results", []))
```

## Summary Checklist

When implementing a new service:
- [ ] Define `.proto` file with custom methods
- [ ] Implement Servicer class with all RPC methods
- [ ] Use `wrap_value()` / `unwrap_value()` for data conversion
- [ ] Set large message size limits for images/videos
- [ ] Add idle timeout logic for GPU services
- [ ] Log errors and return empty Envelope on failure
- [ ] Test standalone before pipeline integration
