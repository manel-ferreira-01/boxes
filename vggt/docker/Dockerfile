# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace
# This argument will be used to identify the
# proto file as well as the python file with the server
# The proto file should have the name ${SERVICE_NAME}.proto
# The python file should have the name ${SERVICE_NAME}_service.py
ARG SERVICE_NAME=simplebox


# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM python:slim-buster AS builder
#FROM python:latest as builder
## Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
#COPY image_generic.proto ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile proto file and remove it
RUN python -m grpc_tools.protoc -I. --python_out=.  --grpc_python_out=. ${SERVICE_NAME}.proto


# -----------------------------
# Stage to generate final image
# -----------------------------

#FROM python:slim-buster
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 
# Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src


# Create non-privileged user and workspace
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

# Install requirements
COPY requirements.txt .
RUN apt update -y && \
    apt install -y pip ffmpeg libsm6 libxext6 &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    # Install headless version of opencv-python for server usage
    # Does not install graphical modules
    # See https://github.com/opencv/opencv-python#installation-and-usage
    pip install -r requirements.txt && \
    rm requirements.txt

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/${SERVICE_NAME}.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file and the utils to workspace
# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# Change to non-privileged user
USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

# -----------------------------
# Specific steps for vggt service
# -----------------------------

COPY --chown=${USER}:${GROUP} ${SRC}/vggt ${WORKSPACE}/vggt
RUN pip install -r ${WORKSPACE}/vggt/requirements.txt

# set $TORCH_HOME for torchhub cache
ENV TORCH_HOME=${WORKSPACE}/.cache
ENV HF_HOME=${WORKSPACE}/.cache
ENV HF_HUB_CACHE=${WORKSPACE}/.cache

#copy vggt-1b.pt to workspace
COPY --chown=${USER}:${GROUP} docker/vggt-1b.pt ${WORKSPACE}/vggt-1b.pt

#print the torchhub cache path
RUN echo "Torch hub cache path: $TORCH_HOME"

CMD ["python3", "service.py"]