# 3D reconstruction with VGGT:
### Building the container:
Check docker directory. Check the requirements listed in 
https://github.com/facebookresearch/vggt.git

Keep the service name as simplebox if you want to keep all files untouched (protos, gprc, protobuf)

```bash
$ docker build --tag sipgisr/vggtgrpc --build-arg SERVICE_NAME=vggt -f docker/Dockerfile .
```

### Data format 

- **Input** : check test notebook: repeated bytes of images.
- **Output** : a dictionary with all data returned by vggt
- dict_keys(['pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images'])

### Launching the container with the service

```bash
$ docker run -it --gpus all -p 8061:8061 sipgisr/vggtgrpc bash -c "python3 service.py"
```
You may want to copy the cache to /home/runner/.cache during build or map volume

