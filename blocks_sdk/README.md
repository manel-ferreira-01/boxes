# blocks_sdk

A dynamic Python SDK to interact with microservice AI boxes via gRPC without needing a priori knowledge of service methods or protobuf schemas.

## Features

- **Zero-A-Priori Knowledge Service Discovery**: Uses gRPC Server Reflection to discover available services, RPC methods (`Process`, `DetectSequence`, `TrackSequence`, `Forward`), and message types on-the-fly across any port.
- **Dynamic Method Calling**: Call any discovered RPC method directly (`client.list_methods()`, `client.DetectSequence(...)`, `client.Forward(...)`) without pre-compiled stubs.
- **Automatic Protobuf Serialization**: Automatically wraps and unwraps native Python objects (images, text, matrices) into Protobuf envelopes and messages.

## Quick Start

```bash
cd /home/manuelf/boxes/blocks_sdk

# Build and start all services
docker compose up -d --build

# Run test suite and dynamic reflection demo
python3 test_client.py
```

## Dynamic On-Demand Usage (No Prior Method Knowledge Required)

```python
from blocks_sdk import Client

# Connect to ANY microservice box on demand
client = Client("localhost:8062")

# Discover all methods exposed by the box
methods = client.list_methods()
print("Discovered RPC methods:", methods)

# Call any discovered method dynamically
response = client.call(
    method=methods[0],
    data={"images": [img_bytes]},
    config_json='{"threshold": 0.5}'
)

# Or invoke directly as a client method:
response = client.DetectSequence(images=[img_bytes], config_json='{"threshold": 0.5}')
```

## Pre-Defined Service Classes

High-level wrappers are also available for convenience:

```python
from blocks_sdk import YOLO, TAPNext, CoTracker, TextEmbedding, VGGT

yolo = YOLO("localhost:8062")
response = yolo.detect([img_bytes], threshold=0.5)

cotracker = CoTracker("localhost:8063")
tracks = cotracker.track_video(video_bytes)
```

## Service Ports (Default Port Mappings)

- `localhost:8061`: TAPNext / VGGT
- `localhost:8062`: YOLO
- `localhost:8063`: CoTracker
- `localhost:8064`: TextEmbedding
