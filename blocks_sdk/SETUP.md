# Setup Instructions

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ for SDK development/testing

## Quick Start

```bash
# Build services (with GPU support for TAPNext)
docker-compose up -d --build

# Test from host machine
python test_client.py
```

## Project Structure

```
blocks_sdk/
├── blocks_sdk/           # Python package
│   ├── __init__.py       # Package exports
│   ├── client.py         # Base gRPC client
│   ├── helpers.py        # Protobuf helpers (wrap/unwrap)
│   ├── protos/           # Compiled protobuf bindings
│   │   ├── pipeline_pb2.py
│   │   └── pipeline_pb2_grpc.py
│   └── services/         # Service-specific clients
│       ├── yolo.py
│       ├── tapnext.py
│       ├── cotracker.py
│       └── text_embedding.py
├── docker-compose.yml    # Service definitions
├── generate_protos.sh    # Generate protobuf files
└── requirements.txt      # Python dependencies
```

## Architecture

### Docker Networking

Services run in a dedicated Docker network with these names:
- `tapnext-service` → port 8061 (mapped to host 8061)
- `yolo-service` → port 8061 (mapped to host 8062)
- `cotracker-service` → port 8061 (mapped to host 8063)
- `text-embedding-service` → port 8061 (mapped to host 8064)

### Client Connection

```python
# From host machine (with port mapping)
yolo = YOLO("localhost:8062")

# From another container on same network  
yolo = YOLO("yolo-service:8061")
```

## Build Custom Images

If you have modified service images:

```bash
docker-compose build yolo-service
docker-compose up -d
```

Or for all services:
```bash
docker-compose build --no-cache
docker-compose up -d
```
