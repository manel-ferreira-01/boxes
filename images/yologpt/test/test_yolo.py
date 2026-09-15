#!/usr/bin/env python3
"""Test script for the YOLOv11 (yologpt) gRPC service (shared envelope).

Connects to a running yolo box and, using the shared ``Process`` RPC:
  1. `detect`  — the two photos from the clip box test folder -> annotated
     JPEGs + a flat JSON detections list
  2. `track`   — two calls with the same image; the tracker state persists
     across calls, so the same object keeps its id
  3. `reset`   — tracker restarts; the first new id is 1 again

Mirrors the style of images/lang_segm/test/test_lang_sam.py.

Run (server already up on :8061):
    python images/yologpt/test/test_yolo.py
    BOX_HOST=10.0.0.5:8061 python images/yologpt/test/test_yolo.py
"""

import json
import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_TEST_DIR, "..", "protos"))

import grpc          # noqa: E402
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
import aux           # noqa: E402

_CLIP_TEST_DIR = os.path.normpath(os.path.join(_TEST_DIR, "..", "..", "clip", "test"))
_OUT_DIR = _TEST_DIR


def load_test_image(name: str) -> bytes:
    """Load a bundled test photo (from the clip box test folder).

    Falls back to a generated image if the file is not present.
    """
    path = os.path.join(_CLIP_TEST_DIR, name)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            data = f.read()
        print(f"loaded {path}: {len(data) / (1024 * 1024):.2f} MB")
        return data

    print(f"warning: {path} not found, generating a placeholder image")
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(7)
    img = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype=np.uint8),
                          "RGB")
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def call_stub(stub, config: dict, images: list | None = None) -> pipeline_pb2.Envelope:
    data = {}
    if images is not None:
        data["images"] = aux.wrap_value(images)
    req = pipeline_pb2.Envelope(config_json=json.dumps(config), data=data)
    return stub.Process(req)


def main() -> int:
    target = os.getenv("BOX_HOST", "localhost:8061")
    print(f"Target: {target}")

    image_files = ["dog.jpg", "car.jpg"]
    images = [load_test_image(n) for n in image_files]

    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
        ],
    )
    stub = pipeline_pb2_grpc.PipelineServiceStub(channel)

    # ------------------------------------------------------------------ detect
    print("\n== detect ==")
    resp = call_stub(stub,
                     {"yolo": {"command": "detect",
                               "parameters": {"conf": 0.5, "iou": 0.7}}},
                     images)
    cfg = json.loads(resp.config_json or "{}")
    section = next((v for v in cfg.values() if isinstance(v, dict)), {})
    status = section.get("status")
    print(f"config: {cfg}")
    if status != "done":
        print(f"FAIL: expected status 'done', got {status!r}")
        return 1

    print(f"declared encoding: {section.get('encoding')}")
    if section.get("encoding") != {"images": "identity", "detections": "json"}:
        print("FAIL: declared encoding mismatch")
        return 1

    ann = list(aux.unwrap_value(resp.data["images"]))
    dets = json.loads(bytes(aux.unwrap_value(resp.data["detections"])))
    print(f"annotated images: {len(ann)}; detections: {len(dets)}")
    for i, a in enumerate(ann):
        out_path = os.path.join(
            _OUT_DIR, f"output_{os.path.splitext(image_files[i])[0]}_yolo.jpg")
        with open(out_path, "wb") as f:
            f.write(bytes(a))
        print(f"  saved {out_path} ({len(bytes(a))} bytes)")
    if dets:
        by_img: dict = {}
        for d in dets:
            by_img.setdefault(d["image_index"], []).append(d)
        for idx, rows in sorted(by_img.items()):
            print(f"  image {idx}: {len(rows)} detection(s), e.g. "
                  f"{rows[0]['class_name']} conf={rows[0]['confidence']:.2f}")
        if any("track_id" in d for d in dets):
            print("FAIL: detect mode must not include track_id")
            return 1
    else:
        print("  (no detections on test photos — OK if the model sees nothing)")

    # ------------------------------------------------------------------- track
    print("\n== track (stateful across calls) ==")
    call_stub(stub, {"yolo": {"command": "reset"}})
    r1 = call_stub(stub, {"yolo": {"command": "track"}}, images[:1])
    ids1 = [d["track_id"] for d in
            json.loads(bytes(aux.unwrap_value(r1.data["detections"])))]
    r2 = call_stub(stub, {"yolo": {"command": "track"}}, images[:1])
    ids2 = [d["track_id"] for d in
            json.loads(bytes(aux.unwrap_value(r2.data["detections"])))]
    print(f"call 1 ids: {ids1}")
    print(f"call 2 ids: {ids2}")
    if ids1 and ids2:
        # A real tracker associates the same object across calls: the same
        # image sent twice must keep the same id (not mint new ones, and not
        # restart at 1 either).
        if ids2 != ids1:
            print("FAIL: same object should keep its track id across calls")
            return 1
    else:
        print("  (no detections on test photo — nothing to track)")

    # ------------------------------------------------------------------- reset
    print("\n== reset ==")
    rr = call_stub(stub, {"yolo": {"command": "reset"}})
    rcfg = json.loads(rr.config_json or "{}")
    rsec = next((v for v in rcfg.values() if isinstance(v, dict)), {})
    print(f"reset -> {rcfg}")
    if rsec.get("status") != "done":
        print("FAIL: reset should report status 'done'")
        return 1

    r3 = call_stub(stub, {"yolo": {"command": "track"}}, images[:1])
    ids3 = [d["track_id"] for d in
            json.loads(bytes(aux.unwrap_value(r3.data["detections"])))]
    print(f"after reset, ids: {ids3}")
    # after reset the tracker is fresh: its first frame's new tracks start at 1
    if ids3 and ids3[0] != 1:
        print("FAIL: track ids should restart at 1 after reset")
        return 1

    channel.close()
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
