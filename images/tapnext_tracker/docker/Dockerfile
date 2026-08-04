ARG WORKSPACE=/workspace
ARG SERVICE_NAME=tapnext

FROM python:slim-buster AS builder
ARG WORKSPACE
ARG SERVICE_NAME
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto

FROM nvidia/cuda:12.2.2-base-ubuntu22.04
ARG WORKSPACE
ARG SERVICE_NAME
ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src

RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

COPY requirements.txt .
RUN apt update -y && \
    apt install -y pip ffmpeg libsm6 libxext6 &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

USER ${USER}

EXPOSE 8061

WORKDIR ${WORKSPACE}

ENV TORCH_HOME=${WORKSPACE}/.cache
ENV HF_HOME=${WORKSPACE}/.cache
ENV HF_HUB_CACHE=${WORKSPACE}/.cache

RUN echo "Torch hub cache path: $TORCH_HOME"

# Download TAPNext checkpoint during build (740MB)
# Download manually: wget -O ckpt.npz https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
# Then mount with: docker run --rm --gpus all -v /path/ckpt.npz:/workspace/bootstapnext_ckpt.npz sipgisr/tapnexttracker
CMD ["python3", "service.py"]