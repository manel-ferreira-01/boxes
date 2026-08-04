import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
import json
import sys
import io

# Copy proto files from vggt/protos (or symlink)
sys.path.append("./protos")
import pipeline_pb2 as tapnext_pb2
import pipeline_pb2_grpc as tapnext_pb2_grpc
from aux import wrap_value, unwrap_value

import numpy as np
import torch
import cv2

# Import from tapnet package (copied during build)
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
        self._device = "cpu"
        self._last_request_time = time.time()
        self._lock = None
        
        # Tracking state maintained across requests
        self.tracking_state = None
        self.all_tracks = {}  # track_id -> list of (frame_idx, x, y) or None
        self.next_track_id = 0
        self.frame_counter = 0
        self.initialized = False
        
        # Load model on background thread to avoid blocking server startup
        import threading
        self._load_event = threading.Event()
        self._loader_thread = threading.Thread(target=self._load_model_async, daemon=True)
        self._loader_thread.start()
        
        # Watchdog to move model back to CPU when idle
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        
        logging.info("TAPNext service initialized (background loading)")
    
    def _load_model_async(self):
        """Background thread that loads TAPNext model."""
        try:
            logging.info("Loading TAPNext model...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            
            model = TAPNext(
                image_size=(256, 256),
                width=768,
                patch_size=(8, 8),
                num_heads=12,
                lru_width=768,
                depth=12,
            )
            
            checkpoint_path = "/workspace/bootstapnext_ckpt.npz"
            if os.path.exists(checkpoint_path):
                logging.info(f"Loading checkpoint from {checkpoint_path}")
                restore_model_from_jax_checkpoint(model, checkpoint_path)
                logging.info("TAPNext model checkpoint loaded successfully")
            else:
                raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
            model.to(device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            
            self._model = model
            logging.info(f"TAPNext model loaded on {device}")
        except Exception as e:
            logging.exception(f"Failed to load TAPNext model: {e}")
        finally:
            self._load_event.set()
    
    def _watchdog_loop(self):
        """Move model back to CPU when idle for IDLE_TIMEOUT seconds."""
        while True:
            time.sleep(30)
            with self._lock or {}:
                if not hasattr(self, '_last_request_time'):
                    continue
                idle_time = time.time() - self._last_request_time
                if idle_time > _IDLE_TIMEOUT and self._device == "cuda":
                    logging.info("Idle timeout: moving model back to CPU")
                    self._model.to("cpu")
                    torch.cuda.empty_cache()
                    self._device = "cpu"
    
    def Process(self, request, context):
        # Wait for model load to complete
        while not self._load_event.is_set():
            time.sleep(0.5)
        
        with self._lock or {}:
            self._last_request_time = time.time()
            
            # Restore to GPU if needed
            if self._device == "cpu" and torch.cuda.is_available() and hasattr(self, '_model'):
                logging.info("Request received: moving model to GPU")
                try:
                    self._model.to("cuda")
                    self._device = "cuda"
                except Exception as e:
                    logging.warning(f"Failed to move to GPU: {e}")
        
        start_time = time.time()
        
        try:
            if not request.config_json:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({"tapnext": {"status": "error", "error": "No config JSON"}})
                )
            
            config = json.loads(request.config_json)
            tapnext_config = config.get("tapnext", {})
            parameters = tapnext_config.get("parameters", {}) or {}
            
            # Check for reset command
            if tapnext_config.get("command") == "reset" or parameters.get("reset"):
                self.reset_tracking()
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({
                        "tapnext": {"status": "done", "action": "reset"}
                    })
                )
            
            # Handle empty request
            if not request.data.get("images"):
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({
                        "tapnext": {"status": "empty_request"}
                    })
                )
            
            # Process images
            image_bytes_list = unwrap_value(request.data["images"])
            if not isinstance(image_bytes_list, list) or len(image_bytes_list) == 0:
                return tapnext_pb2.Envelope(
                    config_json=json.dumps({
                        "tapnext": {"status": "error", "error": "No images in data"}
                    })
                )
            
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
            
            # Serialize outputs
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
                config_json=json.dumps({
                    "tapnext": {"status": "error", "error": str(e)}
                })
            )
    
    def _decode_image(self, img_bytes):
        """Decode image bytes to numpy array."""
        try:
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                from PIL import Image
                import io as pil_io
                img_pil = Image.open(pil_io.BytesIO(img_bytes)).convert("RGB")
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            logging.error(f"Failed to decode image: {e}")
            return None
    
    def _track_frame(self, frame_np, parameters):
        """Track points in a single frame."""
        if frame_np.ndim == 2:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2RGB)
        
        h, w = frame_np.shape[:2]
        
        # Resize to model input size (256x256)
        frame_resized = cv2.resize(frame_np, (256, 256))
        frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
        frame_tensor = frame_tensor.to(self._device)
        
        with torch.no_grad():
            if not self.initialized:
                # First frame: initialize with grid or query points
                grid_size = parameters.get("grid_size", 32)
                
                # Generate query points on grid
                y_coords = np.linspace(0, 1, grid_size)
                x_coords = np.linspace(0, 1, grid_size)
                xx, yy = np.meshgrid(x_coords, y_coords)
                query_points_np = np.stack([xx.flatten(), yy.flatten()], axis=1)  # [N, 2]
                
                # Create query points with time=0
                query_points = []
                for x, y in query_points_np:
                    query_points.append([0.0, x, y])
                query_points_tensor = torch.tensor(query_points).float().unsqueeze(0).to(self._device)
                
                # Run initial inference
                tracks, track_logits, visible_logits, self.tracking_state = self._model(
                    frame_tensor,
                    query_points=query_points_tensor,
                    state=None
                )
                
                # Initialize tracking history
                num_points = tracks.shape[2]
                for i in range(num_points):
                    self.all_tracks[self.next_track_id] = [(0, x.item(), y.item()) if visible_logits[0, 0, i].item() > 0 else (0, None, None)]
                    self.next_track_id += 1
                
                self.initialized = True
            else:
                # Subsequent frames: continue with state preservation
                tracks, track_logits, visible_logits, self.tracking_state = self._model(
                    frame_tensor,
                    query_points=None,
                    state=self.tracking_state
                )
            
            # Convert to numpy and resize coords back to original dimensions
            tracks_np = tracks.cpu().numpy()[0, 0]  # [N, 2]
            visibles_np = visible_logits.cpu().numpy()[0, 0] > 0
            
            # Scale coordinates back to original frame size
            scale_y, scale_x = h / 256.0, w / 256.0
            tracks_np[:, 0] *= scale_x
            tracks_np[:, 1] *= scale_y
            
            return tracks_np, visibles_np
    
    def _serialize_tensor(self, tensor):
        """Serialize torch tensor to bytes."""
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf, pickle_protocol=4)
        return buf.getvalue()
    
    def reset_tracking(self):
        """Reset tracking state for new sequence."""
        self.tracking_state = None
        self.all_tracks = {}
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
