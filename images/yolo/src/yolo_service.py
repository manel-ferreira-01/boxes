"""YOLO box — object detection on images and videos (ultralytics).

A standard shared-envelope box, stateless and clip-shaped:

* ``data["images"]`` — list of image bytes (JPEG/PNG/…); one detection
  record per image.
* ``data["video"]``  — one video file (mp4/avi/webm/mov); the box decodes
  it server-side (OpenCV/ffmpeg), samples ``frame_step``, and returns one
  detection record per sampled frame (with the original frame index).
* ``command: reset`` — accepted as a no-op (the box holds no state).

GPU lifecycle (fleet convention): the model loads on CPU at startup, moves
to CUDA in place on the first request (unless ``parameters.device`` says
otherwise), and a watchdog thread moves it back to CPU after ``_IDLE_TIMEOUT``
seconds of inactivity. Weights are **not** baked into the image: the default
checkpoint (env ``YOLO_WEIGHTS``, default ``yolov8n.pt``) is downloaded/loaded
when the box starts, and ``parameters.weights`` switches the active
checkpoint per call (fresh ones download on first use, into the container
workspace).

Payloads: ``detections`` is JSON (declared via the ``encoding`` contract
key so ``boxes_client`` decodes it); ``annotated`` is a list of JPEG bytes
(declared ``identity``); a video input additionally yields
``annotated_video`` — the annotated frames re-encoded as one MP4 (H.264
when PyAV is available so the browser's ``<video>`` can play it, mp4v
fallback otherwise).
"""

import concurrent.futures as futures
import json
import logging
import os
import sys
import tempfile
import threading
import time

sys.path.append("./protos")
import pipeline_pb2  # noqa: E402
import pipeline_pb2_grpc  # noqa: E402
from aux import wrap_value, unwrap_value  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

_PORT_ENV_VAR = 'PORT'
_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 60  # seconds

_DEFAULT_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")  # runtime download
_DEFAULTS = {
    "conf": 0.25,
    "iou": 0.70,
    "imgsz": 640,
    "max_det": 300,
    "save_annotated": True,
    "frame_step": 1,
    "max_frames": 1024,
}


def _error(message, extra=None):
    section = {"status": "error", "error": message}
    if extra:
        section.update(extra)
    return pipeline_pb2.Envelope(config_json=json.dumps({"yolo": section}))


def _sniff_video_ext(b: bytes) -> str:
    """Best-effort video container extension from magic bytes.

    OpenCV's VideoCapture needs a container it can parse; the extension only
    steers its backend selection, so mis-guessing is low-risk (``.mp4`` is
    the fallback because it covers ISO-BMFF: mp4/mov/m4v all share ``ftyp``).
    """
    if len(b) > 12 and b[4:8] == b"ftyp":
        return ".mp4"
    if b[:4] == b"RIFF" and len(b) > 12 and b[8:12] == b"AVI ":
        return ".avi"
    if b[:4] == b"\x1a\x45\xdf\xa3":          # EBML: webm/mkv
        return ".webm"
    return ".mp4"


class PipelineService(pipeline_pb2_grpc.PipelineServiceServicer):

    def __init__(self):
        import torch  # deferred: ultralytics needs it, the box should not
        from ultralytics import YOLO

        self._torch = torch
        # Weights are NOT baked into the image (keeps the build deterministic
        # and offline-friendly): the default checkpoint downloads at box
        # startup into the container workspace (CWD, writable by the runner
        # user), like clip's startup ViT download. Always loaded to CPU
        # first; moved to GPU lazily on request (below).
        self._models = {}
        self._default_weights = _DEFAULT_WEIGHTS
        self._current_weights = _DEFAULT_WEIGHTS
        self._model = YOLO(_DEFAULT_WEIGHTS)
        self._models[_DEFAULT_WEIGHTS] = self._model
        self._device = "cpu"
        logging.info("YOLO model (%s) loaded on CPU", _DEFAULT_WEIGHTS)

        self._last_request_time = time.time()
        self._lock = threading.Lock()

        # Background thread to monitor idle time.
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    # ------------------------------------------------------------- weights
    def _activate_weights(self, weights):
        """Switch the active checkpoint (under the lock).

        The box stays **stateless**: ``weights`` is resolved *per call*
        (explicit ``parameters.weights``, else the startup default) — the
        cache only avoids re-loading the last-used checkpoints, it never
        leaks state between calls. Inactive cached models are parked on CPU
        (a GPU-resident model is moved down before the switch) so only the
        *active* model can hold VRAM — the watchdog lifecycle stays 1:1
        with the active model. Unknown checkpoints are fetched by
        ultralytics on first use (the download lands in the container
        workspace).
        """
        with self._lock:
            if weights == self._current_weights:
                return
            if self._device.startswith("cuda"):
                logging.info("Switching weights: parking %s on CPU", self._current_weights)
                self._model.to("cpu")
                self._torch.cuda.empty_cache()
                self._device = "cpu"
            if weights not in self._models:
                logging.info("Loading checkpoint %s (first use may download)", weights)
                from ultralytics import YOLO
                self._models[weights] = YOLO(weights)
            self._model = self._models[weights]
            self._current_weights = weights
            logging.info("Active checkpoint: %s (CPU)", weights)

    # ------------------------------------------------------------------ GPU
    def _watchdog_loop(self):
        while True:
            time.sleep(10)  # check every 10s
            with self._lock:
                idle_time = time.time() - self._last_request_time
                if idle_time > _IDLE_TIMEOUT and self._device.startswith("cuda"):
                    logging.info("Idle timeout reached: moving model back to CPU")
                    self._model.to("cpu")
                    self._torch.cuda.empty_cache()
                    self._device = "cpu"

    def _place_model(self, parameters):
        """Move the model (under the lock): explicit ``parameters.device``
        wins; otherwise CUDA if visible. Returns the resulting device."""
        target = str(parameters.get("device") or "").strip().lower()
        if not target and self._torch.cuda.is_available():
            target = "cuda"
        if target and self._device != target:
            logging.info("Moving model to %s", target)
            with self._lock:
                self._model.to(target)
                self._device = target
        return self._device

    # ---------------------------------------------------------------- params
    @staticmethod
    def _num(parameters, key, cast=float):
        if key not in parameters:
            return cast(_DEFAULTS[key])
        try:
            v = cast(parameters[key])
        except (TypeError, ValueError) as e:
            raise ValueError(f"parameters.{key} must be a number, got {parameters[key]!r}") from e
        if v < 0:
            raise ValueError(f"parameters.{key} must be >= 0, got {v}")
        return v

    @staticmethod
    def _boolish(v, key):
        """True/False for JSON booleans and the string forms the webui
        select produces (``"true"`` / ``"false"``)."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(v, (int, float)):
            return bool(v)
        raise ValueError(f"parameters.{key} must be true/false, got {v!r}")

    def _parse_parameters(self, parameters):
        """Validate + coerce user parameters into a clean inference spec."""
        classes = parameters.get("classes")
        if classes is not None:
            if not isinstance(classes, (list, tuple)) or not all(
                    isinstance(c, (int, float, str)) for c in classes):
                raise ValueError("parameters.classes must be a list of class ids")
            try:
                classes = [int(float(c)) for c in classes]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"parameters.classes must be a list of (stringable) ints, got {classes!r}") from e

        return {
            "conf": min(1.0, self._num(parameters, "conf")),
            "iou": min(1.0, self._num(parameters, "iou")),
            "imgsz": int(self._num(parameters, "imgsz", int)),
            "max_det": int(self._num(parameters, "max_det", int)),
            "classes": classes,
            "save_annotated": self._boolish(
                parameters.get("save_annotated", _DEFAULTS["save_annotated"]), "save_annotated"),
            "frame_step": max(1, int(self._num(parameters, "frame_step", int))),
            "max_frames": max(1, int(self._num(parameters, "max_frames", int))),
        }

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def _decode_images(image_bytes_list):
        frames = []
        for i, raw in enumerate(image_bytes_list):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"could not decode image #{i + 1} of "
                                 f"{len(image_bytes_list)} (not a supported raster?)")
            frames.append(frame)
        return frames

    def _decode_video(self, video_bytes, frame_step, max_frames):
        """Server-side decode: (sampled BGR frames, their original indices,
        total frame count, source fps).  The total comes from the container
        metadata (``CAP_PROP_FRAME_COUNT``) when the codec provides it, else
        falls back to the number of frames actually decoded before the
        ``max_frames`` cap."""
        suffix = _sniff_video_ext(video_bytes)
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = tmp.name
            tmp.write(video_bytes)
            tmp.close()

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError(f"cv2.VideoCapture could not open the video "
                                 f"(sniffed container {suffix!r} — unsupported codec?)")

            try:
                meta_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            except (TypeError, ValueError):
                meta_total = 0
            try:
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            except (TypeError, ValueError):
                fps = 0.0
            if not (fps > 0) or fps != fps:      # NaN guard
                fps = 30.0

            frames, indices = [], []
            decoded = 0
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if decoded % frame_step == 0:
                    frames.append(frame)
                    indices.append(decoded)
                decoded += 1
            cap.release()
            return frames, indices, (meta_total if meta_total > 0 else decoded), fps
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _encode_mp4(frames, fps):
        """Encode BGR frames into one MP4, preferring the **browser-playable
        H.264** (PyAV bundles the FFmpeg libraries incl. libx264).  Degrades
        to ``mp4v`` (MPEG-4 Part 2, plays in VLC but NOT in HTML5 <video>)
        when PyAV is missing or the encode fails.  Returns ``(bytes, codec)``
        or ``(None, None)"."""
        if not frames:
            return None, None
        h, w = frames[0].shape[:2]
        # H.264 needs even dimensions — pad by one black pixel when odd.
        if w % 2 or h % 2:
            w2, h2 = w + w % 2, h + h % 2
            frames = [cv2.copyMakeBorder(f, 0, h2 - h, 0, w2 - w,
                                         cv2.BORDER_CONSTANT, value=0)
                      for f in frames]
        else:
            w2, h2 = w, h

        # --- preferred: H.264 via PyAV (browser <video> support) ----------
        try:
            import av
        except ImportError:
            av = None
        if av is not None:
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp_path = tmp.name
                tmp.close()
                rate = max(1, int(round(fps))) if fps and fps > 0 else 30
                container = av.open(tmp_path, mode="w")
                stream = container.add_stream("libx264", rate=rate)
                stream.width, stream.height, stream.pix_fmt = w2, h2, "yuv420p"
                for frame in frames:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for pkt in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                        container.mux(pkt)
                for pkt in stream.encode(None):      # flush
                    container.mux(pkt)
                container.close()
                with open(tmp_path, "rb") as f:
                    data = f.read()
                return (data, "h264") if data else (None, None)
            except (OSError, ValueError) as e:
                logging.warning("h264 encode failed (%s); falling back to mp4v", e)
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        # --- fallback: mp4v via OpenCV (non-browser codecs are fine here) --
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp.name
            tmp.close()
            writer = cv2.VideoWriter(
                tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps if fps and fps > 0 else 30.0, (w2, h2))
            if not writer.isOpened():
                return None, None
            for frame in frames:
                writer.write(frame)
            writer.release()
            with open(tmp_path, "rb") as f:
                data = f.read()
            return (data, "mp4v") if data else (None, None)
        except (OSError, cv2.error):
            logging.exception("annotated_video encoding failed")
            return None, None
        finally:
            try:
                os.unlink(tmp_path)
            except (OSError, NameError):
                pass

    # -------------------------------------------------------------- inference
    def _detect_frame(self, r, frame, frame_index):
        """One frame's Result -> a plain-JSON detection record."""
        xyxy = labels = class_ids = scores = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes):
            names = self._model.names
            xyxy = np.round(boxes.xyxy.cpu().numpy(), 2).tolist()
            class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            labels = [names[c] for c in class_ids]
            scores = np.round(boxes.conf.cpu().numpy(), 4).tolist()
        h, w = frame.shape[:2]
        return {
            "frame_index": frame_index,   # original index in the source (video frame nº / list pos)
            "width": int(w),
            "height": int(h),
            "boxes": xyxy,                # [x1, y1, x2, y2] in input-frame pixels
            "class_ids": class_ids,
            "labels": labels,
            "scores": scores,
        }

    # ---------------------------------------------------------------- Process
    def Process(self, request, context):
        start_time = time.time()

        try:
            if not request.config_json:
                return _error("No config JSON")

            config = json.loads(request.config_json)
            yolo_config = config.get("yolo", {})
            if not isinstance(yolo_config, dict):
                return _error("config section 'yolo' must be an object")
            parameters = yolo_config.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                return _error("parameters must be an object")

            # Stateless box: accept "reset" (client convenience) as a no-op.
            if yolo_config.get("command") == "reset" or parameters.get("reset"):
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({"yolo": {"status": "done", "action": "reset"}}))

            try:
                spec = self._parse_parameters(parameters)
            except ValueError as e:
                return _error(str(e))

            image_bytes_list = unwrap_value(request.data["images"]) if "images" in request.data else None
            if image_bytes_list is not None and not isinstance(image_bytes_list, list):
                return _error("data.images must be a list of image bytes")
            video_bytes = unwrap_value(request.data["video"]) if "video" in request.data else None

            if image_bytes_list and video_bytes:
                return _error("send either data.images or data.video, not both")
            if not image_bytes_list and not video_bytes:
                return pipeline_pb2.Envelope(
                    config_json=json.dumps({"yolo": {"status": "empty_request"}}))

            # Weights: resolved per call — explicit parameters.weights or
            # the startup default (stateless box: no sticky state). Fresh
            # checkpoints download on first use.
            weights = str(parameters.get("weights") or "").strip() \
                or self._default_weights
            try:
                self._activate_weights(weights)
            except Exception as e:
                logging.exception("failed to load weights %r", weights)
                return _error(f"failed to load weights {weights!r}: {e}")

            # GPU lifecycle: CPU at startup -> lazy move on request -> idle
            # watchdog fallback (see _place_model / _watchdog_loop).
            device = self._place_model(parameters)

            source = "images"
            extra_status = {}
            fps = 0.0
            if video_bytes:
                frames, frame_indices, total_frames, fps = self._decode_video(
                    video_bytes, spec["frame_step"], spec["max_frames"])
                source = "video"
                extra_status = {
                    "frames_in_video": total_frames,
                    "frame_step": spec["frame_step"],
                }
                if not frames:
                    return _error("no frames could be decoded from data.video")
            else:
                frames = self._decode_images(image_bytes_list)
                frame_indices = list(range(len(frames)))

            results = self._model(
                frames,
                conf=spec["conf"],
                iou=spec["iou"],
                imgsz=spec["imgsz"],
                max_det=spec["max_det"],
                classes=spec["classes"],
                device=device,
                verbose=False,
            )
            if len(results) != len(frames):
                return _error(f"inference returned {len(results)} results for "
                              f"{len(frames)} frames (internal error)")

            detections = [self._detect_frame(results[i], frames[i], frame_indices[i])
                          for i in range(len(frames))]
            num_detections = sum(len(d["boxes"]) for d in detections)

            response_data = {
                "detections": wrap_value(json.dumps(detections).encode("utf-8")),
            }
            encoding = {"detections": "json"}

            if spec["save_annotated"]:
                annotated_bgr, annotated = [], []
                for r in results:
                    # ``r.plot()`` is BGR: ultralytics draws its annotations
                    # on the *original* frame, which we decoded BGR with
                    # cv2. Keep it BGR — ``cv2.imencode`` and the mp4v
                    # ``VideoWriter`` both expect BGR, and the H.264 path
                    # does its own BGR->RGB conversion before x264.  Converting
                    # *here* (as before) double-swapped channels and put
                    # red/blue the wrong way round in the JPEGs and video.
                    bgr_frame = np.ascontiguousarray(r.plot())
                    annotated_bgr.append(bgr_frame)
                    ok, buf = cv2.imencode(".jpg", bgr_frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        annotated.append(buf.tobytes())
                response_data["annotated"] = wrap_value(annotated)
                encoding["annotated"] = "identity"
                if source == "video":
                    # A video in -> a real video out: the same annotated
                    # frames re-encoded as one mp4 (source fps).  H.264 when
                    # PyAV is available (browser <video>), else mp4v fallback.
                    out_video, out_codec = self._encode_mp4(annotated_bgr, fps)
                    if out_video:
                        response_data["annotated_video"] = wrap_value(out_video)
                        encoding["annotated_video"] = "identity"
                        extra_status["annotated_video_codec"] = out_codec

            return pipeline_pb2.Envelope(
                config_json=json.dumps({
                    "yolo": {
                        "status": "done",
                        "weights": self._current_weights,
                        "runtime": time.time() - start_time,
                        "source": source,
                        "num_frames": len(frames),
                        "frames_sampled": len(frames),
                        "num_detections": num_detections,
                        # Declared payload encoding (generic boxes_client
                        # contract): JSON detection records, raw JPEG bytes.
                        "encoding": encoding,
                        **extra_status,
                    }
                }),
                data=response_data,
            )

        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return _error(str(e))


def get_port():
    """Parse the port where the server should listen.

    Exits the program if the environment variable is not a positive int.

    Returns:
        The port where the server should listen, or None if an error occurred.
    """
    try:
        server_port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if server_port <= 0:
            logging.error('Port should be greater than 0')
            return None
        return server_port
    except ValueError:
        logging.exception('Unable to parse port')
        return None


def run_server(server):
    """Run the given server on the port defined by the environment variables
    or the default port if it is not defined."""
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'''Server started at {target}''')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    import grpc
    import grpc_reflection.v1alpha.reflection as grpc_reflection

    logging.basicConfig(
        format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO)

    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ]
    )

    pipeline_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        pipeline_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
