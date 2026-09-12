# boxes — documentation

A fleet of independent, Dockerized AI inference services ("boxes") speaking one
shared gRPC envelope, called by `boxes_client` or by orchestration-layer boxes.

## Start here

- **[Architecture_Overview](Architecture_Overview.md)** — what a box is, the
  conventions every box follows, GPU memory lifecycle, what's retired (Maestro).
- **[Quick_Start_Guide](Quick_Start_Guide.md)** — run a box, call it, build your
  own from scratch, test it.
- **[gRPC_Services_Reference](gRPC_Services_Reference.md)** — the envelope
  contract: `config_json` shape per box, `data` fields, `results` encoding,
  devices, calling via `boxes_client` or raw stubs.
- **[Docker_Image_Template_Guide](Docker_Image_Template_Guide.md)** — Dockerfile
  templates (CPU / CUDA / YOLO) used by the boxes in this repo.

## Boxes in this repo

| Box | Type | What it does |
|-----|------|--------------|
| clip | GPU | CLIP image/text embeddings |
| tapnext_tracker | GPU | Point tracking with TAPNext (stateful) |
| lang_segm | GPU | Text-guided segmentation (LangSAM) |
| textEmbedding | GPU / CPU | Sentence-BERT text embeddings |
| vggt | GPU | 3D reconstruction from image sequences |
| yologpt | GPU | YOLOv11 detection & tracking |
| opencv_box | GPU / CPU | Optical flow, feature matching, similarity checks |
| folder_wd | CPU | File-watcher: watches a directory, drives boxes, saves outputs |
| gradio_display | CPU | Gradio UI: upload/images, drive boxes, display results |

**The per-box README is the authoritative source for that box's request shape**
(config keys, fields, status, how to decode `results`):
[`images/<name>/README.md`](../images/).

## The client

- [`boxes_client/README.md`](../boxes_client/README.md) — the Python client
  (`Box(ip:port).run(data, config)`), coercion rules, result decoding, `info()`.
