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

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_IDLE_TIMEOUT = 120  # seconds

logging.basicConfig(
    format='[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

class PipelineService(tapnext_pb2_grpc.PipelineServiceServicer):
    
    def __init__(self):
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._last_request_time = time.time()
        self._lock = threading.Lock()
        self._load_event = threading.Event()
        
        self.tracking_state = None
        self.active_tracks = {}
        self.track_histories = {}
        self.next_track_id = 0
        self.frame_counter = 0
        self.initialized = False
        
        self._loader_thread = threading.Thread(target=self._load_model_async, daemon=True)
        self._loader_thread.start()
        
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        
        logging.info("TAPNext service initialized.")
    
    def _load_model_async(self):
        try:
            logging.info("Loading TAPNext model...")
            model = TAPNext(
                image_size=(256, 256),
                width=768,
                patch_size=(8, 8),
                num_heads=12,
                lru_width=768,
                depth=12,
            ).to(self._device)
            
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            
            checkpoint_path = "/workspace/bootstapnext_ckpt.npz"
            if os.path.exists(checkpoint_path):
                restore_model_from_jax_checkpoint(model, checkpoint_path)
                logging.info(f"Loaded checkpoint from {checkpoint_path}")
            else:
                logging.warning(f"Checkpoint not found at {checkpoint_path}")
            
            self._model = model
        except Exception as e:
            logging.exception(f"Failed to load TAPNext model: {e}")
        finally:
            self._load_event.set()
    
    def _watchdog_loop(self):
        while True:
            time.sleep(30)
            with self._lock:
                idle_time = time.time() - self._last_request_time
                if idle_time > _IDLE_TIMEOUT and self._device == "cuda" and self._model is not None:
                    logging.info("Idle timeout: moving model to CPU")
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"

    def Process(self, request, context):
        while not self._load_event.is_set():
            time.sleep(0.1)
        
        with self._lock:
            self._last_request_time = time.time()
            if self._device == "cpu" and torch.cuda.is_available() and self._model is not None:
                logging.info("Request received: restoring model to GPU")
                self._model.to("cuda")
                self._device = "cuda"

        start_time = time.time()
        
        try:
            if not request.config_json:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({"tapnext": {"status": "error", "error": "No config JSON"}})
                )
            
            config = json.loads(request.config_json)
            tapnext_config = config.get("tapnext", {})
            parameters = tapnext_config.get("parameters", {}) or {}
            
            if tapnext_config.get("command") == "reset" or parameters.get("reset"):
                self.reset_tracking()
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({"tapnext": {"status": "done", "action": "reset"}})
                )
            
            if not request.data.get("images"):
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({"tapnext": {"status": "empty_request"}})
                )
            
            image_bytes_list = unwrap_value(request.data["images"])
            if not isinstance(image_bytes_list, list) or len(image_bytes_list) == 0:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({"tapnext": {"status": "error", "error": "No images in data"}})
                )
            
            # Reset state for new sequence processing batch
            self.reset_tracking()
            
            all_tracks = []
            all_visibles = []
            
            for img_bytes in image_bytes_list:
                frame_np = self._decode_image(img_bytes)
                if frame_np is None:
                    continue
                
                tracks, visibles = self._track_frame(frame_np, parameters)
                if tracks is not None:
                    all_tracks.append(tracks)
                    all_visibles.append(visibles)
            
            response_data = {}
            if all_tracks:
                response_data["tracks"] = wrap_value(self._serialize_tensor(torch.tensor(all_tracks)))
                response_data["visibles"] = wrap_value(self._serialize_tensor(torch.tensor(all_visibles)))
            
            return tapnext_pb2.Envelope(
                config_json=json.dumps({
                    "tapnext": {
                        "status": "done",
                        "frames_processed": len(all_tracks),
                        "runtime": time.time() - start_time
                    }
                }),
                data=response_data
            )
        
        except Exception as e:
            logging.exception(f"Error in Process: {e}")
            return tapnext_pb2.Envelope(
                config_json=json.dumps({"tapnext": {"status": "error", "error": str(e)}})
            )

    def _decode_image(self, img_bytes):
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logging.error(f"Failed to decode image: {e}")
            return None

    def _track_frame(self, frame_np, parameters):
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

        current_frame_idx = self.frame_counter
        
        with torch.no_grad():
            use_amp = (self._device == "cuda")
            with torch.amp.autocast(self._device, dtype=torch.float16, enabled=use_amp):
                if not self.initialized:
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
                    tracks, track_logits, visible_logits, self.tracking_state = self._model(
                        video=frame_tensor,
                        query_points=query_points_tensor
                    )
                    
                    num_feats = tracks.shape[2]
                    for i in range(num_feats):
                        tid = self.next_track_id
                        self.next_track_id += 1
                        self.active_tracks[i] = tid
                        vis = visible_logits[0, 0, i].item() > 0
                        pos = tracks[0, 0, i, :2].cpu() if vis else None
                        self.track_histories[tid] = [(current_frame_idx, pos)]
                    
                    self.initialized = True
                else:
                    # Sequential step inference using saved tracking_state
                    tracks, track_logits, visible_logits, self.tracking_state = self._model(
                        video=frame_tensor,
                        state=self.tracking_state
                    )
                    
                    visible = (visible_logits[0, 0] > 0).cpu()
                    for i in range(tracks.shape[2]):
                        tid = self.active_tracks.get(i, None)
                        if tid is not None:
                            pos = tracks[0, 0, i, :2].cpu() if visible[i] else None
                            self.track_histories[tid].append((current_frame_idx, pos))
                
                self.frame_counter += 1
                
                tracks_np = tracks.cpu().numpy()[0, 0].copy() # [N, 2] -> (x, y)
                visibles_np = (visible_logits.cpu().numpy()[0, 0] > 0)

                # 2. Flip column order from (x, y) to (y, x)
                tracks_yx = tracks_np # Index 0 is now Y, Index 1 is now X

                # 3. Calculate scaling factors matching (y, x) -> (orig_h, orig_w)
                scale_y = orig_h / 256.0
                scale_x = orig_w / 256.0

                # 4. Scale Y by scale_y, X by scale_x
                tracks_yx[:, 0] *= scale_y  # Y coordinate (scaled to orig_h)
                tracks_yx[:, 1] *= scale_x  # X coordinate (scaled to orig_w)

                return tracks_yx, visibles_np

    def _serialize_tensor(self, tensor):
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf, pickle_protocol=4)
        return buf.getvalue()

    def reset_tracking(self):
        self.tracking_state = None
        self.active_tracks = {}
        self.track_histories = {}
        self.next_track_id = 0
        self.frame_counter = 0
        self.initialized = False
        logging.info("Tracking state reset")


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
