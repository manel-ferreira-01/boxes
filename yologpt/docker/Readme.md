
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

Dockerfile uses a builder to generate grpc files that are copied to the final container


### 🔨 Build Docker Image

```bash
docker build -t sipgisr/yolov11 -f docker/Dockerfile .    
```
### 🔨 Deploy Service

```bash
docker run --rm  --name yolo  -ti -p 8061:8061 sipgisr/yolov11 
```
