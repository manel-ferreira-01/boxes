# 3D reconstruction with VGGT:
### Building the container:
Check docker directory. Check the requirements listed in 
https://github.com/facebookresearch/vggt.git

Keep the service name as simplebox if you want to keep all files untouched (protos, gprc, protobuf)

```bash
$ docker build --tag sipgisr/vggtGrpc: --build-arg SERVICE_NAME=simplebox -f docker/Dockerfile .
```

### Data format 


- **Input** : a mat file with a dictionary named imgdata with a list of encoded images
- **Output** : a mat file with all data returned by vggt
- dict_keys(['__header__', '__version__', '__globals__', 'pose_enc', 'depth', 'depth_conf', 'world_points', 'world_points_conf', 'images'])

### Launching the container with the service

```bash
$ docker run -p XXXX:8061 -d vggt_name 
```
You may want to copy the cache to /home/runner/.cache during build or map volume

