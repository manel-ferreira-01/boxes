#!/bin/bash
# Generate Python protobuf files

echo "Generating protobuf Python bindings..."

python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    pipeline.proto

echo "Done! Generated:"
ls -la protos/ 2>/dev/null || ls -la pipeline_pb2*.py
