#!/usr/bin/env python3
"""In-process smoke test for the yologpt box (shared envelope, `Process`).

Runs the REAL `yoloservice.PipelineService` in-process behind a real gRPC
server, with a **fake ultralytics model** (no GPU, no weights, no network),
and drives it end-to-end through a real `boxes_client.Box`:

  1. `detect`  — two images -> annotated JPEGs + flat JSON detections
  2. `track`   — two calls, same session -> track_ids persist across calls
  3. `reset`   — tracker counter restarts
  4. error paths — unknown command, empty images, missing/invalid config
  5. contract  — status/encoding conventions, client-side `json`-codec decode

Needs: grpcio, grpcio-reflection, numpy, opencv (imdecode/imencode), zstandard.
Run from the repo (any cwd):

    python3 images/yologpt/test/test_yolo_fake.py
"""

import concurrent.futures as futures
import json
import os
import sys
import time
import types
import numpy as np
import cv2

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.normpath(os.path.join(TEST_DIR, "..", "src"))
REPO_ROOT = os.path.normpath(os.path.join(TEST_DIR, "..", "..", ".."))

# ------------------------- fake ultralytics -------------------------------

_RESETS = {"count": 0}
_TRACKER_CALLS = []   # every tracker-callback invocation lands here (regression probe)


class _XYRow:
    def __init__(self, i):
        self._i = i

    def tolist(self):
        i = self._i
        return [4 + i, 8 + i, 60 + i, 72 + i]


class _XY:
    """Supports ``b.xyxy[0].tolist()`` like a real ultralytics box."""

    def __init__(self, i):
        self._i = i

    def __getitem__(self, k):
        return _XYRow(self._i)


class _FakeBox:
    def __init__(self, i, yolo):
        self.i = i
        self.xyxy = _XY(i)
        self.conf = np.float32(0.9 - 0.1 * i)
        self.cls = np.int64(0)
        self.id = np.int64(yolo.next_track_id()) if yolo.mode == "track" else None


class _FakeResult:
    def __init__(self, img, yolo):
        self.orig_img = img
        self.boxes = [_FakeBox(i, yolo) for i in (0, 1)]

    def plot(self, img=None):
        out = np.ascontiguousarray(img.copy())
        cv2.rectangle(out, (4, 8), (60, 72), (255, 0, 0), 2)
        return out


class _FakeTracker:
    def reset(self):
        _RESETS["count"] += 1


class _FakePredictor:
    trackers = [_FakeTracker()]


class _FakeYOLO:
    """Mimics `ultralytics.YOLO`: .names, .predictor, __call__/track,
    and a *stateful* track-id counter (reset via predictor.trackers[0].reset())."""

    def __init__(self, weights):
        assert weights == "yolo11n.pt"
        self.names = {0: "person", 2: "car"}
        self.predictor = _FakePredictor()
        self._track_id = 0
        self.mode = "detect"
        self.last_params = None
        # event registry, like the real model: track() installs the tracker
        # callbacks here *globally*, and every predict runs them
        self.callbacks = {"on_predict_start": [], "on_predict_postprocess_end": []}
        # wire reset() to our counter (the box calls reset() on the tracker)
        self.predictor.trackers[0].reset = self._reset_counter

    def _reset_counter(self):
        _RESETS["count"] += 1
        self._track_id = 0

    def next_track_id(self) -> int:
        tid = self._track_id
        self._track_id += 1
        return tid

    def _install_tracker(self, persist):
        """Mimic `model.track()`: register the tracker callbacks globally."""
        from functools import partial
        from ultralytics.trackers import track as tt
        self.callbacks["on_predict_start"] = [
            partial(tt.on_predict_start, persist=persist)]
        self.callbacks["on_predict_postprocess_end"] = [
            partial(tt.on_predict_postprocess_end, persist=persist)]

    def _run_callbacks(self):
        for ev in ("on_predict_start", "on_predict_postprocess_end"):
            for cb in self.callbacks[ev]:
                cb(self.predictor)

    def __call__(self, img, **params):
        self.mode = "detect"
        self.last_params = params
        self._run_callbacks()      # real predictor runs the events too
        return [_FakeResult(img, self)]

    def track(self, source=None, persist=False, **params):
        assert persist
        self._install_tracker(persist)
        self.mode = "track"
        self.last_params = params
        self._run_callbacks()
        return [_FakeResult(source, self)]


def _install_fake_ultralytics():
    mod = types.ModuleType("ultralytics")
    mod.YOLO = _FakeYOLO
    trackers = types.ModuleType("ultralytics.trackers")
    track_mod = types.ModuleType("ultralytics.trackers.track")

    def on_predict_start(predictor, persist=False):
        _TRACKER_CALLS.append("start")

    def on_predict_postprocess_end(predictor, persist=False):
        _TRACKER_CALLS.append("postprocess")

    track_mod.on_predict_start = on_predict_start
    track_mod.on_predict_postprocess_end = on_predict_postprocess_end
    mod.trackers = trackers
    trackers.track = track_mod
    sys.modules["ultralytics"] = mod
    sys.modules["ultralytics.trackers"] = trackers
    sys.modules["ultralytics.trackers.track"] = track_mod


def _jpeg(h=96, w=96, seed=0) -> bytes:
    """A real, decodable JPEG (random noise)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


# ------------------------------ the test ----------------------------------

def main():
    _install_fake_ultralytics()

    # yoloservice resolves "../protos" relative to the CWD (container layout:
    # WORKDIR /workspace with yoloservice.py + protos flat), so run from src/.
    os.chdir(SRC_DIR)

    sys.path.insert(0, SRC_DIR)                 # yoloservice + its protos
    sys.path.insert(0, REPO_ROOT + "/boxes_client/src")
    import grpc  # noqa: E402
    import yoloservice  # noqa: E402
    from boxes_client import Box  # noqa: E402

    from grpc_reflection.v1alpha.reflection import (  # noqa: E402
        enable_server_reflection)
    import pipeline_pb2  # noqa: E402
    import pipeline_pb2_grpc  # noqa: E402

    # start the box in-process on a free port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4),
                         options=[("grpc.max_send_message_length", -1),
                                  ("grpc.max_receive_message_length", -1)])
    svc = yoloservice.PipelineService()
    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(svc, server)
    enable_server_reflection(
        (pipeline_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,),
        server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    addr = f"127.0.0.1:{port}"
    print(f"fake yolo box listening on {addr}")

    b = Box(addr, config_key="yolo")
    passes = 0

    def check(name, cond, extra=""):
        nonlocal passes
        if not cond:
            print(f"FAIL: {name} {extra}")
            sys.exit(1)
        passes += 1
        print(f"  ok: {name}")

    # ----------------------------------------------------------------- 1. detect
    print("== detect ==")
    res = b.run(data={"images": [_jpeg(seed=1), _jpeg(seed=2)]},
                config={"yolo": {"command": "detect",
                                 "parameters": {"conf": 0.5}}})
    check("status done", res.config["yolo"]["status"] == "done", res.config)
    check("command echoed", res.config["yolo"]["command"] == "detect")
    check("declared encoding", res.encoding ==
          {"images": "identity", "detections": "json"})
    check("2 annotated images (raw bytes)",
          isinstance(res.images, list) and len(res.images) == 2
          and all(isinstance(x, (bytes, bytearray)) for x in res.images))
    check("detections auto-decoded (json codec) -> list",
          isinstance(res.detections, list) and len(res.detections) == 4)
    d0 = res.detections[0]
    check("flat dets carry image_index/bbox/class",
          d0["image_index"] == 0 and len(d0["bbox"]) == 4
          and d0["class_name"] == "person" and "track_id" not in d0)
    check("num_images/num_detections",
          res.config["yolo"]["num_images"] == 2
          and res.config["yolo"]["num_detections"] == 4)
    check("parameters reached the model",
          svc.model.last_params == {"conf": 0.5})

    # ----------------------------------------------------------------- 2. track
    print("== track (stateful across calls) ==")
    r1 = b.run(data={"images": [_jpeg(seed=3)]},
               config={"yolo": {"command": "track"}})
    r2 = b.run(data={"images": [_jpeg(seed=4)]},
               config={"yolo": {"command": "track"}})
    ids1 = [d["track_id"] for d in r1.detections]
    ids2 = [d["track_id"] for d in r2.detections]
    check("track dets have track_id", all("track_id" in d for d in r1.detections))
    check("ids persist/continue across calls", ids1 == [0, 1] and ids2 == [2, 3],
          f"call1={ids1} call2={ids2}")
    check("status done on both",
          r1.config["yolo"]["status"] == "done"
          and r2.config["yolo"]["status"] == "done")

    # ------------------------------------------ detect never touches the tracker
    print("== detect after track must not run the tracker (regression) ==")
    before = len(_TRACKER_CALLS)
    rd = b.run(data={"images": [_jpeg(seed=20)]},
               config={"yolo": {"command": "detect"}})
    check("detect status done", rd.config["yolo"]["status"] == "done", rd.config)
    check("detect did not invoke tracker callbacks",
          len(_TRACKER_CALLS) == before, _TRACKER_CALLS)
    r2b = b.run(data={"images": [_jpeg(seed=21)]},
                config={"yolo": {"command": "track"}})
    ids2b = [d["track_id"] for d in r2b.detections]
    check("track ids continue after an intervening detect",
          ids2b == [4, 5], f"got {ids2b}")

    # ----------------------------------------------------------------- 3. reset
    print("== reset ==")
    r3 = b.run(config={"yolo": {"command": "reset"}})
    check("reset -> status done", r3.config["yolo"]["status"] == "done",
          r3.config)
    check("reset cleared data (status-only response)", r3.fields == {})
    r4 = b.run(data={"images": [_jpeg(seed=5)]},
               config={"yolo": {"command": "track"}})
    check("track ids restart after reset",
          [d["track_id"] for d in r4.detections] == [0, 1])

    # --------------------------------------------------------------- 4. errors
    print("== error paths ==")
    bad = b.run(data={"images": [_jpeg(seed=6)]},
                config={"yolo": {"command": "warp9"}})
    check("unknown command -> status error",
          bad.config["yolo"]["status"] == "error")
    check("unknown command lists known",
          bad.config["yolo"]["known"] == ["detect", "track", "reset"])

    empty = b.run(config={"yolo": {"command": "detect"}})
    check("no images -> empty_request",
          empty.config["yolo"]["status"] == "empty_request", empty.config)

    noconf = b.call(yoloservice.pipeline_pb2.Envelope())
    check("missing config -> error",
          noconf.config["yolo"]["status"] == "error")

    # legacy aliases: no section at all still means 'detect'
    leg = b.run(data={"images": [_jpeg(seed=7)]}, config={})
    check("legacy: no section -> detect",
          leg.config["yolo"]["status"] == "done"
          and leg.config["yolo"]["command"] == "detect")

    legacy2 = b.run(data={"images": [_jpeg(seed=8)]},
                    config={"YOLO": {"command": "detect"}})
    check("legacy: 'YOLO' section alias",
          legacy2.config["yolo"]["status"] == "done")

    # single image sent as ONE bytes value (`b`, not `bb`) still works
    req = pipeline_pb2.Envelope(config_json=json.dumps({"yolo": {"command": "detect"}}))
    req.data["images"].CopyFrom(yoloservice.wrap_value(_jpeg(seed=11)))
    single = b.call(req)
    check("single `b` image accepted",
          single.config["yolo"]["status"] == "done"
          and single.config["yolo"]["num_images"] == 1,
          single.config)

    # legacy stream countdown reset
    b.run(config={"yolo": {"command": "reset"}})
    s = b.run(data={"images": [_jpeg(seed=9)]},
              config={"yolo": {"command": "track", "stream": 0}})
    check("legacy: stream=0 still triggers end-of-sequence reset",
          s.config["yolo"].get("action") == "stream_reset")

    # ------------------------------------------------------- 5. info + reflection
    info = b.info()
    check("reflection probe sees pipeline.PipelineService",
          info["reachable"] and info["reflection"] is True and
          info["methods"] == ["Process"], info)

    b.close()
    server.stop(0)
    print(f"\nALL {passes} CHECKS PASSED (fake ultralytics, in-process gRPC, "
          "driven through boxes_client)")


if __name__ == "__main__":
    main()
