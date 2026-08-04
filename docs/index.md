# Documentation Index

Welcome to the pipeline system documentation.

## Getting Started

- [Pipeline Architecture Overview](Pipeline_Architecture_Overview.md)
- [Quick Start Guide](Quick_Start_Guide.md)

## Core Concepts

- [gRPC Services Reference](gRPC_Services_Reference.md)
- [Pipeline Configuration Reference](Pipeline_Configuration_Reference.md)

## Templates & Examples

- [Docker Image Template Guide](Docker_Image_Template_Guide.md)

## Service Examples in /images/

| Service | Type | Description |
|---------|------|-------------|
| opencv_box | CPU | Feature matching, optical flow, similarity check |
| vggt | GPU | 3D reconstruction from multiple images |
| yologpt | GPU | YOLOv11 detection tracking |
| cotracker | GPU | Video motion tracking with CoTracker |
| gradio_display | CPU | Web UI for user interaction |
| textEmbedding | GPU | Text embeddings |

## Development Workflow

1. **Create service** - Use templates to build a new Docker image
2. **Test standalone** - Run `docker run` to test your service
3. **Add to pipeline** - Configure Maestro in /home/manuelf/boxes/pipelines/
4. **Deploy** - Start complete pipeline with docker-compose

## Need Help?

- Check existing pipelines for examples: /home/manuelf/boxes/pipelines/*/config.yaml
- Review service code: /home/manuelf/boxes/images/*_service.py

