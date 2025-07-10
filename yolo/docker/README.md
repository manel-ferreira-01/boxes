# OpenCV gRPC Docker integration


## Overview

This directory provides Dockerfiles to be able to run the gRPC service with Docker.
The docker image already has all the project dependencies installed.
Also, it is already built with the necessary sources for the gRPC service to run.


## Usage

In order to use the images, execute the following command:

```shell
$ docker run --rm -it -p 8061:8061 --mount type=bind,source=<path to optional host directory>,target=/workspace/external user/simplebox 
```

NOTE: The `<path to optional host directory>` must be the absolute path to some directory needed to run the service (it is optional).


## Building the image

In this repository, we define multiple gRPC services. Keep or rename the .py and protobuf files to reflect the name of your service (ex replace all simplebox references by the name you chose).

In order to build the image for a specific service, execute the respective command *(from the repository root directory)*:

### SimpleBox Service 

```shell
$ docker build --tag sipgisr/name_of_your_service: --build-arg SERVICE_NAME=name_of_your_service -f docker/Dockerfile .
```
