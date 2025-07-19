This _Box_ provides a gRPC service for the [YoloV11 from Ultalytics](https://github.com/ultralytics/ultralytics) model for object detection.

The service is fully compliant with the [AI4Europe](http://ai4europe.eu) platform definitions and can be launched as a docker service or as a _standalone_ python script. 


---

## 🚀 Launching the service using docker (recommended)

The service can be deployed as a docker service downloadable from docker hub :

```shell
$ docker run --rm -p 8061:8061 --ipc=host sipgisr/yolov11:latest
```
The service will download the model from the ultralytics repository. If you have a local copy then mount the model from host (here we use another port as an example):


```bash
docker run -p XXXX:8061 -v $(pwd)/yolo11n.pt:/workspace/yolo11n.pt sipgisr/yolov11
```

To generate the container locally see the [docker folder](docker)

---

### 📁 Yolo Service Project Structure


```
yolo_grpc/
├── [protos](protos)
    ├──yolo.proto
├── src
    ├── yolo_server.py
├── test
├── docker
    ├── Dockerfile
├── requirements.txt
└── yolo11n.pt   # YOLOv11 model file
```

## 📦 Requirements

Install required Python packages:

```bash
pip install grpcio grpcio-tools opencv-python ultralytics
```

Generate gRPC Python files from `yolo.proto`:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. yolo.proto
```

---

## 🧾 Service Protobuf Interface

The Yolov11 service receives one image and detection parameters in **YOLORequest** message, the image encoded as a jpeg/png binary file and the threshold as a flot. It returns the image with detected labels and a json string with the detections in message **YOLOResponse** . 

### `YOLORequest`

| Field                  | Type   | Description                                             |
|------------------------|--------|---------------------------------------------------------|
| `image`                | `bytes`| Image data encoded as JPEG/PNG                          |
| `confidence_threshold` | `float`| Filter threshold for confidence (range 0.0 to 1.0)      |

### `YOLOResponse`

| Field            | Type     | Description                                  |
|------------------|----------|----------------------------------------------|
| `labeled_image`  | `bytes`  | JPEG image with bounding boxes drawn         |
| `detections_json`| `string` | JSON array with detected object metadata     |

Example detection JSON:
```json
{
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.89,
  "class_id": 0,
  "class_name": "person"
}
```
