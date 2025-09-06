#!/bin/bash

# Usage: ./generate_grpc.sh your_file.proto

# Exit on error
set -e

# Check if argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 your_file.proto"
  exit 1
fi

PROTO_FILE="$1"

# Check if file exists
if [ ! -f "$PROTO_FILE" ]; then
  echo "Error: File '$PROTO_FILE' not found."
  exit 1
fi

# Get filename without extension
BASENAME=$(basename "$PROTO_FILE" .proto)

echo "Generating gRPC Python files for '$PROTO_FILE'..."

python -m grpc_tools.protoc -I. \
  --python_out=. \
  --grpc_python_out=. \
  "$PROTO_FILE"

echo "Generated: ${BASENAME}_pb2.py and ${BASENAME}_pb2_grpc.py"