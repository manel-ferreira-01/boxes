# Gradio Interface Component


## Overview

This directory provides Dockerfiles to be able to run the gRPC service with Docker.
The docker image already has all the project dependencies installed.
Also, it is already built with the necessary sources for the gRPC service to run.


## Usage

In order to use the images, execute the following command:

```shell
$ docker run --rm -it -p 8061:8061 -p 7860:7860  sipgisr/gradiosipg
```

Open a web browser and type localhost:7860 . You should see Gradio components

## Building the image

In this repository, we define multiple gRPC services. Keep or rename the .py and protobuf files to reflect the name of your service (ex replace all simplebox references by the name you chose).

In order to build the image for a specific service, execute the respective command *(from the repository root directory)*:

### SimpleBox Service 

```shell
$ docker build --tag sipgisr/name_of_your_service: --build-arg SERVICE_NAME=name_of_your_service -f docker/Dockerfile .
```
