# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace
# This argument will be used to identify the
# proto file as well as the python file with the server
# The proto file should have the name ${SERVICE_NAME}.proto
# The python file should have the name ${SERVICE_NAME}_service.py
ARG SERVICE_NAME=yolosimplebox

# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------


FROM python:3.11.10-slim-bookworm as builder

## Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio==1.71.0 grpcio-tools==1.71.0  protobuf==5.29.4


COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
#COPY image_generic.proto ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile proto file and remove it
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ${SERVICE_NAME}.proto


# -----------------------------
# Stage to generate final image
# -----------------------------
FROM python:3.11.10-slim-bookworm

#FROM python:slim-buster
# Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget  \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-privileged user and workspace
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

#YOLO V12 specifics - no need because it's on ultralytics
#RUN wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl


# Install requirements
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm requirements.txt

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/${SERVICE_NAME}.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# Change to non-privileged user
USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

CMD ["python", "service.py"]