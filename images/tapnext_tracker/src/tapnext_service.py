import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
import json
import sys
import io
import threading

sys.path.append("./protos")
import pipeline_pb2 as tapnext_pb2
import pipeline_pb2_grpc as tapnext_pb2_grpc
from aux import wrap_value, unwrap_value

import numpy as np
import torch
import cv2


def build_observation_matrix(tracks_list):
    """
    Build Tomasi-Kanade observation matrix P from tracked points.

    Args:
        tracks_list: list of [num_points, 2] arrays (y, x coordinates) per frame

    Returns:
        P: observation matrix of shape (2 * num_frames, num_points)
           Row order: [x1..xN, y1..yN] for all frames concatenated
    """
    if not tracks_list:
        return None

    all_tracks = np.stack(tracks_list)  # [F, N, 2]
    F, N, _ = all_tracks.shape

    P = np.zeros((2 * F, N), dtype=np.float32)

    for f in range(F):
        y_coords = all_tracks[f, :, 0]  # Y coordinates
        x_coords = all_tracks[f, :, 1]  # X coordinates

        P[2*f, :] = x_coords
        P[2*f + 1, :] = y_coords

    return P


def default_model_factory(device):
    """
    Load the real TAPNext model. Kept as a factory (and imported lazily) so the
    session layer can be exercised in-process with a stub model (see test/).
    """
    from tapnet.tapnext.tapnext_torch import TAPNext
    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

    logging.info("Loading TAPNext model...")
    model = TAPNext(
        image_size=(256, 256),
        width=768,
        patch_size=(8, 8),
        num_heads=12,
        lru_width=768,
        depth=12,
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    checkpoint_path = "/workspace/bootstapnext_ckpt.npz"
    if os.path.exists(checkpoint_path):
        restore_model_from_jax_checkpoint(model, checkpoint_path)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logging.warning(f"Checkpoint not found at {checkpoint_path}")

    return model


_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 120  # seconds of global inactivity before the model parks on CPU
_DEFAULT_SESSION = "default"
# Time-of-day a given student session may sit idle before it (and its GPU state)
# is reaped. Default: keep sessions forever (classroom mode). Set e.g. 1800 to
# reclaim per-session VRAM on shared GPUs; 0 disables reaping entirely.
_SESSION_TTL = float(os.getenv("TAPNEXT_SESSION_TTL", "0"))

logging.basicConfig(
    format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


class Session:
    """
    All per-user tracking state. One Session per `session_id`.

    This is exactly the block of globals the service used to keep on `self`;
    moving it here (plus the per-session lock) is what makes the box
    multi-tenant: the model and device lifecycle stay global, everything a
    user can observe lives on their Session only.
    """

    __slots__ = (
        "tracking_state", "active_tracks", "track_histories", "next_track_id",
        "frame_counter", "initialized", "full_tracking_data",
        "accumulated_tracks", "accumulated_visibles", "last_used", "lock",
    )

    def __init__(self):
        self.tracking_state = None
        self.active_tracks = {}
        self.track_histories = {}
        self.next_track_id = 0
        self.frame_counter = 0
        self.initialized = False
        self.full_tracking_data = []   # Tracks accumulated across requests (observation matrix)
        self.accumulated_tracks = []   # Tracks accumulated for responses (y, x coordinates)
        self.accumulated_visibles = []
        self.last_used = time.time()
        self.lock = threading.Lock()   # L2: serializes requests WITHIN this session


class PipelineService(tapnext_pb2_grpc.PipelineServiceServicer):
    """
    Multi-session TAPNext box.

    Lock ordering (global, no exceptions — prevents deadlocks):
      L1  self._sessions_lock   — the sessions dict, session create/delete
      L2  Session.lock          — one per session, held for the full request
      L3  self._device_lock     — the model's CPU<->CUDA move only
    A lock may only be taken if no higher-numbered lock is held.
    """

    def __init__(self, model_factory=None, watchdog_interval=30.0, session_ttl=_SESSION_TTL):
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_factory = model_factory or default_model_factory
        self._last_request_time = time.time()

        self._sessions = {}                    # sid -> Session
        self._sessions_lock = threading.Lock() # L1
        self._device_lock = threading.Lock()   # L3
        self._session_ttl = session_ttl
        self._watchdog_interval = watchdog_interval
        self._load_event = threading.Event()

        self._loader_thread = threading.Thread(target=self._load_model_async, daemon=True)
        self._loader_thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

        logging.info("TAPNext service initialized (multi-session).")

    # ------------------------------------------------------------------ setup

    def _load_model_async(self):
        try:
            self._model = self._model_factory(self._device)
        except Exception as e:
            logging.exception(f"Failed to load TAPNext model: {e}")
        finally:
            self._load_event.set()

    def _get_session(self, sid):
        """Fetch-or-create the Session for `sid` (L1 only)."""
        with self._sessions_lock:
            sess = self._sessions.get(sid)
            if sess is None:
                sess = Session()
                self._sessions[sid] = sess
                logging.info(f"Session created: {sid} (active sessions: {len(self._sessions)})")
            sess.last_used = time.time()
            return sess

    def _reset_session(self, sess):
        """Reset one session's tracking state. Caller holds no lock (or L1)."""
        sess.tracking_state = None
        sess.active_tracks = {}
        sess.track_histories = {}
        sess.next_track_id = 0
        sess.frame_counter = 0
        sess.initialized = False
        sess.full_tracking_data = []
        sess.accumulated_tracks = []
        sess.accumulated_visibles = []
        sess.last_used = time.time()
        logging.info("Tracking session state reset")

    # ------------------------------------------------------------- devicelife

    def _promote_to_gpu(self):
        """Move the shared model back to GPU. Call while holding L2 for this
        session (never take L1 here — that would invert the lock order)."""
        if self._device == "cpu" and torch.cuda.is_available() and self._model is not None:
            with self._device_lock:  # L3 under L2 — legal ordering
                logging.info("Request received: restoring model to GPU")
                self._model.to("cuda")
                self._device = "cuda"

    def _watchdog_loop(self):
        while True:
            time.sleep(self._watchdog_interval)
            try:
                self._reap_idle_sessions()
                self._park_model_if_idle()
            except Exception:
                logging.exception("Watchdog cycle failed")

    def _reap_idle_sessions(self):
        """Drop sessions idle beyond the TTL (frees their per-session state).
        A session with a request in flight is skipped this cycle (L2 busy)."""
        if not self._session_ttl or self._session_ttl <= 0:
            return
        now = time.time()
        with self._sessions_lock:  # L1
            for sid, sess in list(self._sessions.items()):
                if now - sess.last_used <= self._session_ttl:
                    continue
                if not sess.lock.acquire(timeout=0.05):
                    continue  # in flight — leave it for the next cycle
                try:
                    del self._sessions[sid]
                    logging.info(
                        f"Reaped idle session {sid} (idle {now - sess.last_used:.0f}s)")
                finally:
                    sess.lock.release()  # L2 released before L1

    def _park_model_if_idle(self):
        """Park the (expensive, shared) model on CPU after global idle.
        Session state tensors may remain on GPU; they are the small per-session
        cost and are reaped separately by TTL. Skipped while any session is
        computing (its L2 is held)."""
        if self._device != "cuda":
            return
        if time.time() - self._last_request_time <= _IDLE_TIMEOUT:
            return
        held = []
        with self._sessions_lock:  # L1
            for sess in self._sessions.values():
                if not sess.lock.acquire(timeout=0.05):
                    for s in held:
                        s.lock.release()
                    return  # someone is mid-inference; retry next cycle
                held.append(sess)  # L2 under L1 — legal ordering
            try:
                logging.info(
                    f"Parking model to CPU after _IDLE_TIMEOUT "
                    f"({len(held)} session(s) keep their state on GPU)")
                with self._device_lock:  # L3 under L2 under L1 — legal
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"
            finally:
                for s in held:
                    s.lock.release()

    # ----------------------------------------------------------------- public

    def Process(self, request, context):
        while not self._load_event.is_set():
            time.sleep(0.1)

        try:
            if not request.config_json:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({"tapnext": {"status": "error", "error": "No config JSON"}})
                )

            config = json.loads(request.config_json)
            tapnext_config = config.get("tapnext", {}) or {}
            sid = tapnext_config.get("session_id") or _DEFAULT_SESSION
            parameters = tapnext_config.get("parameters", {}) or {}

            command = tapnext_config.get("command")

            if command == "reset" or parameters.get("reset"):
                # Reset is scoped to THIS session — other sessions are untouched.
                sess = self._get_session(sid)
                with sess.lock:
                    self._reset_session(sess)
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "done", "action": "reset", "session": sid}})
                )

            if command == "list":
                # Operator view: active sessions only (state content is never
                # disclosed). Anyone who can reach the box can enumerate sids.
                with self._sessions_lock:
                    now = time.time()
                    listing = [
                        {
                            "session": kid,
                            "frames_processed": len(k.accumulated_tracks),
                            "num_tracks": len(k.active_tracks),
                            "idle_seconds": round(now - k.last_used, 1),
                        }
                        for kid, k in self._sessions.items()
                    ]
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "done", "action": "list", "sessions": listing}})
                )

            if not request.data.get("images"):
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "empty_request", "session": sid}})
                )

            image_bytes_list = unwrap_value(request.data["images"])
            if not isinstance(image_bytes_list, list) or len(image_bytes_list) == 0:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps(
                        {"tapnext": {"status": "error", "error": "No images in data",
                                     "session": sid}})
                )

            start_time = time.time()

            sess = self._get_session(sid)
            with sess.lock:  # L2: this session's requests are serialized here
                self._last_request_time = time.time()
                self._promote_to_gpu()

                for img_bytes in image_bytes_list:
                    frame_np = self._decode_image(img_bytes)
                    if frame_np is None:
                        continue

                    tracks, visibles = self._track_frame(frame_np, parameters, sess)
                    if tracks is not None:
                        sess.accumulated_tracks.append(tracks)
                        sess.accumulated_visibles.append(visibles)

                        # Accumulate for full observation matrix across requests
                        sess.full_tracking_data.append((tracks.copy(), visibles.copy()))

                response_data = {}
                if sess.accumulated_tracks:
                    tracks_tensor = torch.stack([torch.from_numpy(t) for t in sess.accumulated_tracks])
                    visibles_tensor = torch.stack([torch.from_numpy(v) for v in sess.accumulated_visibles])
                    response_data["tracks"] = wrap_value(self._serialize_tensor(tracks_tensor))
                    response_data["visibles"] = wrap_value(self._serialize_tensor(visibles_tensor))

                    # Build observation matrix from full accumulated tracking data
                    P = build_observation_matrix([t for t, _ in sess.full_tracking_data])
                    if P is not None:
                        response_data["observation_matrix"] = wrap_value(
                            self._serialize_tensor(torch.tensor(P)))

                logging.info(
                    f"session={sid} frames={len(sess.accumulated_tracks)} "
                    f"runtime={time.time() - start_time:.3f}s "
                    f"active_sessions={self._count_sessions()}")

                return tapnext_pb2.Envelope(
                    config_json=json.dumps({
                        "tapnext": {
                            "status": "done",
                            "session": sid,
                            "frames_processed": len(sess.accumulated_tracks),
                            "runtime": time.time() - start_time,
                            "num_points": sess.accumulated_tracks[0].shape[0] if sess.accumulated_tracks else 0,
                            # Declared payload encoding (generic boxes_client contract):
                            # all tensor responses are torch.save()-format bytes.
                            "encoding": {
                                "tracks": "torch",
                                "visibles": "torch",
                                "observation_matrix": "torch",
                            }
                        }
                    }),
                    data=response_data
                )

        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return tapnext_pb2.Envelope(
                config_json=json.dumps({"tapnext": {"status": "error", "error": str(e)}})
            )

    def _count_sessions(self):
        with self._sessions_lock:
            return len(self._sessions)

    # ------------------------------------------------------------- internals

    def _decode_image(self, img_bytes):
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logging.error(f"Failed to decode image: {e}")
            return None

    def _track_frame(self, frame_np, parameters, sess):
        if frame_np.ndim == 2:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
        else:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = frame_np.shape[:2]

        # 1. Resize frame to 256x256
        frame_resized = cv2.resize(frame_np, (256, 256))

        # 2. Prepare tensor: [B=1, T=1, H=256, W=256, C=3]
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).to(self._device)

        current_frame_idx = sess.frame_counter

        with torch.no_grad():
            use_amp = (self._device == "cuda")
            with torch.amp.autocast(self._device, dtype=torch.float16, enabled=use_amp):
                if not sess.initialized:
                    grid_size = parameters.get("grid_size", 32)

                    # FIX: Correct coordinate ordering [t, x, y] for PyTorch grid_sample
                    x_coords = np.linspace(10.0, 246.0, grid_size)
                    y_coords = np.linspace(10.0, 246.0, grid_size)
                    xx, yy = np.meshgrid(x_coords, y_coords, indexing='xy')

                    query_points = []
                    for x, y in zip(xx.flatten(), yy.flatten()):
                        query_points.append([0.0, float(x), float(y)])

                    query_points_tensor = torch.tensor(
                        query_points, dtype=torch.float32
                    ).unsqueeze(0).to(self._device) # [1, N, 3]

                    # TAPNext initialization
                    tracks, track_logits, visible_logits, sess.tracking_state = self._model(
                        video=frame_tensor,
                        query_points=query_points_tensor
                    )

                    num_feats = tracks.shape[2]
                    for i in range(num_feats):
                        tid = sess.next_track_id
                        sess.next_track_id += 1
                        sess.active_tracks[i] = tid
                        vis = visible_logits[0, 0, i].item() > 0
                        pos = tracks[0, 0, i, :2].cpu() if vis else None
                        sess.track_histories[tid] = [(current_frame_idx, pos)]

                    sess.initialized = True
                else:
                    # Sequential step inference using saved tracking state
                    tracks, track_logits, visible_logits, sess.tracking_state = self._model(
                        video=frame_tensor,
                        state=sess.tracking_state
                    )

                    visible = (visible_logits[0, 0] > 0).cpu()
                    for i in range(tracks.shape[2]):
                        tid = sess.active_tracks.get(i, None)
                        if tid is not None:
                            pos = tracks[0, 0, i, :2].cpu() if visible[i] else None
                            sess.track_histories[tid].append((current_frame_idx, pos))

                sess.frame_counter += 1

                tracks_np = tracks.cpu().numpy()[0, 0].copy() # [N, 2] -> (x, y)
                visibles_np = (visible_logits.cpu().numpy()[0, 0] > 0)

                # Flip column order to (y, x), scaled back to original resolution
                tracks_yx = tracks_np # Index 0 is Y, Index 1 is X
                scale_y = orig_h / 256.0
                scale_x = orig_w / 256.0
                tracks_yx[:, 0] *= scale_y  # Y coordinate (scaled to orig_h)
                tracks_yx[:, 1] *= scale_x  # X coordinate (scaled to orig_w)

                return tracks_yx, visibles_np

    def _serialize_tensor(self, tensor):
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf, pickle_protocol=4)
        return buf.getvalue()


def get_port():
    try:
        port = int(os.getenv('PORT', _PORT_DEFAULT))
        if port <= 0:
            logging.error('Port must be positive')
            return None
        return port
    except ValueError:
        logging.exception('Invalid port value')
        return None


def run_server(server):
    port = get_port()
    if not port:
        return

    target = f'[::]:{port}'
    server.add_insecure_port(target)
    server.start()
    logging.info(f'Server started at {target}')

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
        ]
    )

    tapnext_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        tapnext_pb2.DESCRIPTOR.services_by_name['PipelineService'].full_name,
        grpc_reflection.SERVICE_NAME
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
