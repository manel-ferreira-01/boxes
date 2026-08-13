# Quick Start Guide

## 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 2. Build Docker Images

```bash
docker compose up -d --build
```

This will build and start all services:
- TAPNext (GPU, port 8061)
- YOLO (port 8062)  
- CoTracker (port 8063)
- TextEmbedding (port 8064)

## 3. Test the SDK

### Option A: Test from host machine

```python
from blocks_sdk import YOLO, TAPNext

# Connect to services running in Docker
yolo = YOLO("localhost:8062")
tapnext = TAPNext("localhost:8061")

print(f"YOLO service ready at: {yolo.address}")
```

### Option B: Test from within Docker

```bash
docker compose run --rm client_test python test_client.py
```

## 4. Use in Your Project

Put this in your Python project:

```python
import sys
sys.path.insert(0, '/path/to/blocks_sdk')

from blocks_sdk import YOLO

yolo = YOLO("localhost:8062")
response = yolo.detect([image_bytes])
```

Or install as a package:
```bash
pip install -e /path/to/blocks_sdk
```
