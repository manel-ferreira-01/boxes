import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
import json
import sys
import io
import time

# Proto imports
sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2
import pipeline_pb2_grpc as folder_wd_pb2_grpc
from aux import wrap_value, unwrap_value

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = 'PORT'

from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np


# ---------------------------------------------
# Helper: serialize NumPy arrays as .npy bytes
# ---------------------------------------------
def np_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def pad_and_stack(arrays, pad_value=0.0):
    """Pad a list of arrays to the same shape and stack into a tensor."""
    if not arrays:
        return np.zeros((0, 0))
    max_shape = np.max([np.array(a.shape) for a in arrays], axis=0)
    padded = []
    for a in arrays:
        pad_width = [(0, int(m - s)) for s, m in zip(a.shape, max_shape)]
        padded.append(np.pad(a, pad_width, mode='constant', constant_values=pad_value))
    return np.stack(padded, axis=0)


# ---------------------------------------------
# Service Definition
# ---------------------------------------------
class PipelineService(folder_wd_pb2_grpc.PipelineServiceServicer):
    def __init__(self):
        self.prev_frame = None
        self.prev_points = None
        self.frame_counter = 0


    def Process(self, request, context):
        """Process request: extract features and optionally match."""

        #start timer
        start_time = time.time()

        # --- parse parameters safely ---
        try:
            parameters = json.loads(request.config_json)["opencv"]["parameters"]
        except Exception:
            logging.warning("Invalid or missing parameters in config_json. Using defaults.")
            parameters = {}

        feature_extractor = parameters.get("feature_extractor", "SIFT").upper()
        ratio_thresh = parameters.get("ratio_thresh", 0.75)
        max_keypoints = parameters.get("max_keypoints", 500)

        # --- decode images ---
        imgs_in = []
        try:
            for image_bytes in unwrap_value(request.data.get("images", [])):
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    imgs_in.append(img)
            if not imgs_in:
                return folder_wd_pb2.Envelope(
                    config_json=json.dumps({"error": "No valid input images"})
                )
        except Exception:
            return folder_wd_pb2.Envelope(
                config_json=json.dumps({"error": "Failed to decode input images"})
            )

        # Detector setup
        if feature_extractor == "SIFT":
            detector = cv2.SIFT_create(nfeatures=max_keypoints)
        elif feature_extractor == "ORB":
            detector = cv2.ORB_create(nfeatures=max_keypoints)
        else:
            return folder_wd_pb2.Envelope(
                config_json=json.dumps({"error": f"Unsupported extractor {feature_extractor}"})
            )

        keypoints_list, descriptors_list = [], []
        for img in imgs_in:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kps, desc = detector.detectAndCompute(gray, None)
            desc = np.zeros((0, 128), np.float32) if desc is None else desc.astype(np.float32)
            keypoints_list.append(cv2.KeyPoint_convert(kps))
            descriptors_list.append(desc)

        # Pack into tensors
        keypoints_tensor = pad_and_stack(keypoints_list, pad_value=0.0)  # (N, max_K, 2)
        descriptors_tensor = pad_and_stack(descriptors_list, pad_value=0.0)  # (N, max_D, 128)

        # --- not a pair of images ---
        if len(imgs_in) != 2:
            result = {
                "keypoints": wrap_value(np_to_bytes(keypoints_tensor)),
                "descriptors": wrap_value(np_to_bytes(descriptors_tensor))
            }
        else:
            # --- Two : FLANN + F-matrix ---
            descA, descB = descriptors_list[0], descriptors_list[1]

            if feature_extractor == "SIFT":
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)

            knn_matches = flann.knnMatch(descA, descB, k=2)
            good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
            logging.info(f"Good matches after ratio test: {len(good_matches)}")

            if len(good_matches) >= 8:
                ptsA = np.float32([keypoints_list[0][m.queryIdx] for m in good_matches])
                ptsB = np.float32([keypoints_list[1][m.trainIdx] for m in good_matches])
                F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 1.5, 0.999)
                mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(good_matches), bool)
                inliersA, inliersB = ptsA[mask], ptsB[mask]
            else:
                F, inliersA, inliersB = np.zeros((0, 0)), np.zeros((0, 2)), np.zeros((0, 2))

            # Serialize batched tensors
            result = {
                "keypoints": wrap_value(np_to_bytes(keypoints_tensor)),
                "descriptors": wrap_value(np_to_bytes(descriptors_tensor)), # overflows gRPC limit
                "matches_inliers_a": wrap_value(np_to_bytes(inliersA)),
                "matches_inliers_b": wrap_value(np_to_bytes(inliersB)),
                "fundamental_matrix": wrap_value(np_to_bytes(F))
            }

        #print size in MB of each result item
        for k, v in result.items():
            logging.info(f"{k}: {len(unwrap_value(v)) / (1024 * 1024):.2f} MB")

        out_json = {
            "status": "success",
            "runtime": time.time() - start_time,
            "timestamp": time.time()
        }

        return folder_wd_pb2.Envelope(
            data=result,
            config_json=json.dumps(out_json)
        )
    
    def similarity_check(self, request, context):
        start_time = time.time()

        try:
            parameters = json.loads(request.config_json)["opencv"]["parameters"]
        except Exception:
            #logging.warning("Invalid or missing parameters in config_json. Using defaults.")
            parameters = {}

        # --- Common parameters ---
        self.blur_kernel = parameters.get("blur_kernel", 5)

        # --- SSIM params ---
        self.ssim_thresh = parameters.get("ssim_thresh", 0.90)

        # --- Lucas–Kanade params ---
        self.motion_thresh = parameters.get("motion_thresh", 1.5)  # px displacement
        self.max_corners = parameters.get("max_corners", 200)
        self.quality_level = parameters.get("quality_level", 0.01)
        self.min_distance = parameters.get("min_distance", 5)
        self.block_size = parameters.get("block_size", 7)

        # --- decode image ---
        try:
            # Unwrap bytes list safely (some services send [[]], some [b'...'])
            img_list = unwrap_value(request.data.get("images", []))
            if not img_list:
                raise ValueError("Empty image list received.")
            image_bytes = img_list[-1]

            # Ensure it's a numpy uint8 buffer
            nparr = np.frombuffer(image_bytes, dtype=np.uint8)

            # Try decoding using OpenCV first
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Fallback: handle 4-channel PNGs or decoding errors
            if img is None:
                # Sometimes OpenCV fails with 4-channel PNG (transparency)
                logging.warning("cv2.imdecode failed, trying PIL fallback...")
                from PIL import Image
                import io
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            if img is None:
                raise ValueError("Both OpenCV and PIL failed to decode image.")

        except Exception as e:
            #logging.error(f"Failed to decode input image ({type(e).__name__}): {e}")
            return folder_wd_pb2.Envelope(
                config_json=json.dumps({"error": "Failed to decode input image"})
            )
            
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        # --- First frame initialization ---
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_points = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size
            )
            return folder_wd_pb2.Envelope(
                config_json=json.dumps({
                    "status": "ready",
                    "changed": True,
                    "metric": 0.0,
                    "runtime": 0.0
                }),
                data={"images": wrap_value([image_bytes])}
            )

        # =================================================================
        # --- TOGGLE: choose algorithm here ---
        # =================================================================
        if 0:  # <-- flip to 1 to use SSIM instead of Lucas–Kanade
            # === SSIM SIMILARITY CHECK ===
            try:
                score = ssim(gray, self.prev_frame)
                changed = score < self.ssim_thresh
                metric_val = float(score)
                metric_name = "ssim"
            except Exception as e:
                logging.error(f"SSIM computation failed: {e}")
                changed, metric_val, metric_name = False, 0.0, "ssim_error"

        else:
            # === LUCAS–KANADE MOTION DETECTION ===
            try:
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame, gray, self.prev_points, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )

                good_new = next_points[status == 1] if next_points is not None else np.zeros((0, 2))
                good_old = self.prev_points[status == 1] if self.prev_points is not None else np.zeros((0, 2))

                if len(good_new) == 0:
                    mean_motion = 0.0
                else:
                    motion_vectors = good_new - good_old
                    displacements = np.linalg.norm(motion_vectors, axis=1)
                    mean_motion = float(np.mean(displacements))

                changed = mean_motion > self.motion_thresh
                metric_val = mean_motion
                metric_name = "motion"

                # update tracked features
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=self.block_size
                )

            except Exception as e:
                logging.error(f"Optical flow computation failed: {e}")
                changed, metric_val, metric_name = False, 0.0, "motion_error"

        if changed:
            self.prev_frame = gray
        else:
            # Keep comparing against the last "changed" frame
            logging.info("No change — keeping previous reference frame.")


        # --- Build response ---
        out_json = {
            "status": "success",
            "metric_type": metric_name,
            "metric": metric_val,
            "changed": bool(changed),
            "runtime": time.time() - start_time
        }

        return folder_wd_pb2.Envelope(
            config_json=json.dumps(out_json),
            data={"images": wrap_value([image_bytes])} if changed else None
        )

# ---------------------------------------------
# Server setup
# ---------------------------------------------
def get_port():
    try:
        port = int(os.getenv(_PORT_ENV_VAR, _PORT_DEFAULT))
        if port <= 0:
            logging.error("Port must be positive")
            return None
        return port
    except ValueError:
        logging.exception("Invalid port value")
        return None

def run_server(server):
    port = get_port()
    if not port:
        return
    target = f"[::]:{port}"
    server.add_insecure_port(target)
    server.start()
    logging.info(f"Server started at {target}")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    logging.basicConfig(
        format="[ %(levelname)s ] %(asctime)s (%(module)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(),
        options=[('grpc.max_send_message_length',-1), 
               ('grp.max_receive_message_length',-1),
               ('grpc.max_message_length', -1)],
    )
    folder_wd_pb2_grpc.add_PipelineServiceServicer_to_server(PipelineService(), server)

    service_names = (
        folder_wd_pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
        grpc_reflection.SERVICE_NAME,
    )
    grpc_reflection.enable_server_reflection(service_names, server)

    run_server(server)
