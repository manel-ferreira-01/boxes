"""yologpt — YOLOv11 detection & tracking box on the shared envelope.

One contract RPC (`pipeline.PipelineService.Process`), dispatched on the
``command`` in the ``yolo`` config section (the tapnext pattern):

    {"yolo": {"command": "detect" | "track" | "reset",
              "parameters": {…}}}        # parameters -> YOLO kwargs (conf, iou, …)

data in:  ``images`` — list of JPEG/PNG bytes (a single image is a 1-element list)
data out: ``images``     — annotated JPEGs (one per input)
          ``detections`` — one FLAT, ordered list of dicts:
              {"image_index": i, "bbox": [x1, y1, x2, y2], "confidence": c,
               "class_id": k, "class_name": "…", "track_id": n (track mode only)}
          (declared ``encoding``: {"images": "identity", "detections": "json"})

Status follows the fleet convention:
    {"yolo": {"status": "done" | "empty_request" | "error",
              "error": …(on error), "runtime": …, "num_images": N, …}}

Legacy shapes stay accepted (see ``resolve_section``): a ``YOLO`` section, a
flat top-level config, and the old ``"stream": 0`` end-of-sequence reset.
"""

import concurrent.futures as futures
import logging
import os
import time
import json
import sys

sys.path.append("../protos")
import pipeline_pb2
import pipeline_pb2_grpc
from aux import wrap_value, unwrap_value

import cv2
import numpy as np
from ultralytics import YOLO

try:  # ultralytics >= 8.4 layout (callbacks live in trackers/track.py)
    from ultralytics.trackers.track import (
        on_predict_start as _trk_start,
        on_predict_postprocess_end as _trk_post)
    _TRACKER_CALLBACKS = (_trk_start, _trk_post)
except ImportError:  # older ultralytics layouts: fall back to no filtering
    _TRACKER_CALLBACKS = ()

_TRK_EVENTS = ("on_predict_start", "on_predict_postprocess_end")

_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

_COMMANDS = ("detect", "track", "reset")


def _decode_image(img_bytes) -> np.ndarray:
    """JPEG/PNG bytes -> RGB ndarray (the BGR->RGB flip the old service did)."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image (not JPEG/PNG?)")
    return img[..., (2, 1, 0)]


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("could not encode annotated image as JPEG")
    return buf.tobytes()


def _detections_for_image(idx: int, results, names: dict, track: bool) -> list:
    """One input image's YOLO result -> flat detection dicts."""
    out = []
    for r in results:
        for b in r.boxes:
            det = {
                "image_index": idx,
                "bbox": b.xyxy[0].tolist(),
                "confidence": float(b.conf),
                "class_id": int(b.cls),
                "class_name": names.get(int(b.cls), f"class_{int(b.cls)}"),
            }
            if track:
                det["track_id"] = int(b.id.item()) if b.id is not None else -1
            out.append(det)
    return out


def _detect(model, images: list, parameters: dict):
    """Stateless: run detection on each image. -> (annotated_jpegs, flat_dets)

    The tracker callbacks are registered *globally on the model* by the first
    ``track()`` call, so a plain predict would also drive them — without
    suspending them here, every ``detect`` after the first ``track`` would
    advance/pollute the tracker state. They are held off for the duration of
    this call and restored afterwards.
    """
    names = model.names
    saved = {ev: list(model.callbacks.get(ev, [])) for ev in _TRK_EVENTS}
    for ev, cbs in saved.items():
        model.callbacks[ev] = [cb for cb in cbs
                               if getattr(cb, "func", None) not in _TRACKER_CALLBACKS]
    annotated, dets = [], []
    try:
        for i, img_bytes in enumerate(images):
            img = _decode_image(img_bytes)
            results = model(img, **parameters)
            annotated.append(_encode_jpeg(
                results[0].plot(img=np.ascontiguousarray(results[0].orig_img))))
            dets.extend(_detections_for_image(i, results, names, track=False))
    finally:
        model.callbacks.update(saved)
    return annotated, dets


def _track(model, images: list, parameters: dict):
    """Stateful: run tracking across the sequence (persist=True keeps the box
    tracker state, so successive calls continue the same sequence)."""
    names = model.names
    annotated, dets = [], []
    for i, img_bytes in enumerate(images):
        img = _decode_image(img_bytes)
        results = model.track(source=img, persist=True, **parameters)
        annotated.append(_encode_jpeg(results[0].plot(
            img=np.ascontiguousarray(results[0].orig_img))))
        dets.extend(_detections_for_image(i, results, names, track=True))
    return annotated, dets


class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):
    """The shared contract: one ``Process`` RPC, ``command`` dispatch."""

    def __init__(self):
        self.model = YOLO("yolo11n.pt")  # Ensure this is YOLOv11
        logging.info("YOLOv11 model loaded.")

    # ------------------------------------------------------------------ helpers
    def _ok(self, command: str, images: list, detections: list,
            runtime: float, extra: dict | None = None) -> "pipeline_pb2.Envelope":
        section = {
            "status": "done",
            "command": command,
            "runtime": runtime,
            "num_images": len(images),
            "num_detections": len(detections),
            # Declared payload encoding (boxes_client contract, see docs/CODECS.md):
            # images are raw JPEG bytes, detections is a UTF-8 JSON document.
            "encoding": {"images": "identity", "detections": "json"},
        }
        if extra:
            section.update(extra)
        return pipeline_pb2.Envelope(
            data={
                "images": wrap_value(images),
                "detections": wrap_value(json.dumps(detections).encode("utf-8")),
            },
            config_json=json.dumps({"yolo": section}),
        )

    def _status(self, section: dict) -> "pipeline_pb2.Envelope":
        return pipeline_pb2.Envelope(config_json=json.dumps({"yolo": section}))

    @staticmethod
    def resolve_section(config: dict) -> dict:
        """Pick the config section: ``yolo`` (new) -> ``YOLO`` (legacy name)
        -> flat top-level (legacy pipeline format) -> ``{}``."""
        for key in ("yolo", "YOLO"):
            value = config.get(key)
            if isinstance(value, dict):
                return value
        if any(k in config for k in ("command", "parameters", "stream")):
            return config
        return {}

    # -------------------------------------------------------------------- RPC
    def Process(self, request, context):
        start_time = time.time()
        try:
            if not request.config_json:
                return self._status({"status": "error", "error": "No config JSON"})
            try:
                config = json.loads(request.config_json)
            except json.JSONDecodeError:
                return self._status({"status": "error",
                                     "error": "config_json is not valid JSON"})

            section = self.resolve_section(config)
            parameters = section.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                return self._status({"status": "error",
                                     "error": "parameters must be an object"})

            # -- command: explicit value wins; legacy heuristics otherwise --
            command = section.get("command")
            if command is None:
                command = "track" if "stream" in section else "detect"
            if command not in _COMMANDS:
                return self._status({
                    "status": "error",
                    "error": f"unknown command {command!r}",
                    "known": list(_COMMANDS),
                })

            # -- reset: the fleet-wide state-clear convention ----------------
            if command == "reset":
                self._reset_tracker()
                return self._status({"status": "done", "action": "reset"})

            # -- images --------------------------------------------------------
            images_val = request.data.get("images")
            img_list = unwrap_value(images_val) if images_val is not None else []
            if isinstance(img_list, (bytes, bytearray, memoryview)):
                img_list = [bytes(img_list)]      # single image sent as one `b`
            if not img_list:
                return self._status({"status": "empty_request"})

            if command == "track":
                annotated, dets = _track(self.model, img_list, parameters)
            else:
                annotated, dets = _detect(self.model, img_list, parameters)

            extra = {}
            # Legacy: the old TrackSequence reset the tracker when the
            # stream countdown reached 0 -- keep that behaviour.
            if command == "track" and section.get("stream") == 0:
                self._reset_tracker()
                extra["action"] = "stream_reset"

            return self._ok(command, annotated, dets,
                            time.time() - start_time, extra or None)

        except Exception as e:
            logging.exception(f"[yolo Process] {e}")
            return self._status({"status": "error", "error": str(e)})

    def _reset_tracker(self):
        """Reset the YOLO tracker state (id counter, track history)."""
        try:
            if hasattr(self.model.predictor, "trackers"):
                self.model.predictor.trackers[0].reset()
                logging.info("YOLO tracker reset successfully.")
        except Exception as e:
            logging.error(f"Failed to reset YOLO tracker: {e}")


def get_port():
    """Parse the port where the server should listen.

    Exits the program if the environment variable is not a positive int.

    Returns:
        The port where the server should listen, or None if the port is invalid.
    """
    try:
        server_port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if server_port <= 0:
            logging.error("Port should be greater than 0")
            return None
        return server_port
    except ValueError:
        logging.exception("Unable to parse port")
        return None


def run_server(server):
    """Run the given server on the port defined by the env vars."""
    port = get_port()
    if not port:
        return

    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")


if __name__ == "__main__":
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[('grpc.max_send_message_length', -1),
                 ('grpc.max_receive_message_length', -1)],
    )

    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(
        PipelineService(), server)

    # Add reflection
    service_names = (
        pipeline_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
    server.wait_for_termination()
