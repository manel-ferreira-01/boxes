
## 🐳 Deploying YOLOv11 gRPC Server with Docker

### 📁 Project Structure

```
yolo_grpc/
├── protos
    ├──yolo.proto
├── src
    ├── yolo_server.py
├── test
├── docker
    ├── Dockerfile
├── requirements.txt
└── yolo11n.pt   # YOLOv11 model file
```

### 📝 requirements.txt

```
grpcio
grpcio-tools
ultralytics
opencv-python
```

### 🐋 Dockerfile

```Dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. yolo.proto

EXPOSE 8061

CMD ["python", "yolo_server.py"]
```

### 🔨 Build Docker Image

```bash
docker build -t yolov11-grpc .
```
