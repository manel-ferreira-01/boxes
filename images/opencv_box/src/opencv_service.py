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
import torch


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
        # LightGlue models (lazy initialized)
        self._lg_extractor = None
        self._lg_matcher = None
        self._lg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_lightglue(self, extractor_type="superpoint", max_kpts=2048):
        """Initialize LightGlue models."""
        if self._lg_extractor is None or self._lg_matcher is None:
            from lightglue import LightGlue, SuperPoint, DISK
            try:
                if extractor_type.lower() == "superpoint" or not extractor_type:
                    self._lg_extractor = SuperPoint(max_num_keypoints=max_kpts).eval().to(self._lg_device)
                    self._lg_matcher = LightGlue(features='superpoint').eval().to(self._lg_device)
                elif extractor_type.lower() == "disk":
                    self._lg_extractor = DISK(max_num_keypoints=max_kpts).eval().to(self._lg_device)
                    self._lg_matcher = LightGlue(features='disk').eval().to(self._lg_device)
                logging.info(f"LightGlue initialized: {extractor_type} on {self._lg_device}")
            except Exception as e:
                logging.error(f"LightGlue init failed: {e}")
                raise

    def _lightglue_process(self, imgs_in, max_kpts):
        """Process images with LightGlue matching."""
        self._init_lightglue("superpoint", max_kpts)
        
        # Helper to format image tensor for SuperPoint [1, C, H, W]
        def img2tensor(img_bgr):
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self._lg_device)
        
        feats0 = self._lg_extractor.extract(img2tensor(imgs_in[0]))
        feats1 = self._lg_extractor.extract(img2tensor(imgs_in[1]))
        
        # Match features
        matches_res = self._lg_matcher({'image0': feats0, 'image1': feats1})
        
        # Safely unpack keypoints (removing batch dim if present)
        kps0_t = feats0['keypoints']
        if isinstance(kps0_t, list):
            kps0_t = kps0_t[0]
        elif kps0_t.ndim == 3:
            kps0_t = kps0_t[0]

        kps1_t = feats1['keypoints']
        if isinstance(kps1_t, list):
            kps1_t = kps1_t[0]
        elif kps1_t.ndim == 3:
            kps1_t = kps1_t[0]

        kps0 = kps0_t.detach().cpu().numpy()
        kps1 = kps1_t.detach().cpu().numpy()

        # Safely unpack matches (handling list vs Tensor)
        raw_matches = matches_res['matches']
        if isinstance(raw_matches, list):
            matches_t = raw_matches[0]
        elif raw_matches.ndim == 3:
            matches_t = raw_matches[0]
        else:
            matches_t = raw_matches

        matches = matches_t.detach().cpu().numpy()

        pts_a = np.zeros((0, 2), dtype=np.float32)
        pts_b = np.zeros((0, 2), dtype=np.float32)

        if len(matches) > 0:
            m_idx0 = matches[:, 0]
            m_idx1 = matches[:, 1]
            
            # Mask valid keypoint index matches
            valid = (m_idx0 < len(kps0)) & (m_idx1 < len(kps1))
            pts_a = kps0[m_idx0[valid]]
            pts_b = kps1[m_idx1[valid]]

        # Fundamental matrix via RANSAC
        F_mat = np.zeros((0, 0), dtype=np.float32)
        inliers_a = np.zeros((0, 2), dtype=np.float32)
        inliers_b = np.zeros((0, 2), dtype=np.float32)

        if len(pts_a) >= 8:
            F_calc, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 1.0, 0.99)
            if F_calc is not None and mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                F_mat = F_calc
                inliers_a = pts_a[inlier_mask]
                inliers_b = pts_b[inlier_mask]

        # Stack keypoints into padded array shape (2, max_N, 2)
        all_kpts = [kps0[:max_kpts], kps1[:max_kpts]]
        max_len = max(len(k) for k in all_kpts) if any(len(k) > 0 for k in all_kpts) else 1
        
        padded_kpts = []
        for k in all_kpts:
            if len(k) < max_len:
                pad = np.pad(k, ((0, max_len - len(k)), (0, 0)), mode='constant')
                padded_kpts.append(pad)
            else:
                padded_kpts.append(k)
                
        keyp_tensor = np.stack(padded_kpts, axis=0)
        desc_tensor = np.zeros((2, max_len, 256), dtype=np.float32)

        return {
            "keypoints": wrap_value(np_to_bytes(keyp_tensor)),
            "descriptors": wrap_value(np_to_bytes(desc_tensor)),
            "matches_inliers_a": wrap_value(np_to_bytes(inliers_a)),
            "matches_inliers_b": wrap_value(np_to_bytes(inliers_b)),
            "fundamental_matrix": wrap_value(np_to_bytes(F_mat))
        }

    def _parse_extractor(self, fx_param):
        """Parse feature extractor parameter."""
        if not fx_param:
            return ("SIFT", False)
        fx_upper = fx_param.upper()
        if 'SUPERPOINT' in fx_upper or 'DISK' in fx_upper or 'LIGHTGLUE' in fx_upper:
            return ('superpoint', True)
        return (fx_upper, False)

    def Process(self, request, context):
        start_time = time.time()
        if not request.config_json:
            return folder_wd_pb2.Envelope()

        try:
            parameters = json.loads(request.config_json)["opencv"]["parameters"]
        except Exception:
            parameters = {}

        fx_param = parameters.get("feature_extractor", "SIFT")
        extractor_name, use_lightglue = self._parse_extractor(fx_param)
        
        ratio_thresh = parameters.get("ratio_thresh", 0.75)
        max_keypoints = parameters.get("max_keypoints", 2048 if use_lightglue else 500)

        # Decode images
        imgs_in = []
        try:
            for image_bytes in unwrap_value(request.data.get("images", [])):
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    imgs_in.append(img)
            if not imgs_in:
                return folder_wd_pb2.Envelope()
        except Exception:
            return folder_wd_pb2.Envelope()

        # Execute LightGlue path
        if use_lightglue and len(imgs_in) == 2:
            try:
                result = self._lightglue_process(imgs_in, max_keypoints)
                out_json = {
                    "status": "success",
                    "matcher": f"LightGlue ({extractor_name})",
                    "runtime": time.time() - start_time,
                    "timestamp": time.time()
                }
                return folder_wd_pb2.Envelope(data=result, config_json=json.dumps(out_json))
            except Exception as e:
                logging.error(f"LightGlue execution failed: {e}")

        # Execute OpenCV SIFT/ORB path
        if extractor_name == "SIFT":
            detector = cv2.SIFT_create(nfeatures=max_keypoints)
        elif extractor_name == "ORB":
            detector = cv2.ORB_create(nfeatures=max_keypoints)
        else:
            return folder_wd_pb2.Envelope()

        keypoints_list, descriptors_list = [], []
        for img in imgs_in:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kps, desc = detector.detectAndCompute(gray, None)
            desc = np.zeros((0, 128), np.float32) if desc is None else desc.astype(np.float32)
            keypoints_list.append(cv2.KeyPoint_convert(kps))
            descriptors_list.append(desc)

        keypoints_tensor = pad_and_stack(keypoints_list, pad_value=0.0)
        descriptors_tensor = pad_and_stack(descriptors_list, pad_value=0.0)

        if len(imgs_in) != 2:
            result = {
                "keypoints": wrap_value(np_to_bytes(keypoints_tensor)),
                "descriptors": wrap_value(np_to_bytes(descriptors_tensor))
            }
        else:
            descA, descB = descriptors_list[0], descriptors_list[1]
            if extractor_name == "SIFT":
                flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            else:
                flann = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=50))

            knn_matches = flann.knnMatch(descA, descB, k=2)
            good_matches = [m for m, n in knn_matches if m.distance < ratio_thresh * n.distance]

            if len(good_matches) >= 8:
                ptsA = np.float32([keypoints_list[0][m.queryIdx] for m in good_matches])
                ptsB = np.float32([keypoints_list[1][m.trainIdx] for m in good_matches])
                F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC, 1.5, 0.999)
                mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(good_matches), bool)
                inliersA, inliersB = ptsA[mask], ptsB[mask]
            else:
                F, inliersA, inliersB = np.zeros((0, 0)), np.zeros((0, 2)), np.zeros((0, 2))

            result = {
                "keypoints": wrap_value(np_to_bytes(keypoints_tensor)),
                "descriptors": wrap_value(np_to_bytes(descriptors_tensor)),
                "matches_inliers_a": wrap_value(np_to_bytes(inliersA)),
                "matches_inliers_b": wrap_value(np_to_bytes(inliersB)),
                "fundamental_matrix": wrap_value(np_to_bytes(F))
            }

        out_json = {
            "status": "success",
            "matcher": "FLANN",
            "runtime": time.time() - start_time,
            "timestamp": time.time()
        }

        return folder_wd_pb2.Envelope(
            data=result,
            config_json=json.dumps(out_json)
        )
    
    def similarity_check(self, request, context):
        start_time = time.time()

        if not request.config_json: # is not even an empty string
            return folder_wd_pb2.Envelope()

        try:
            parameters = json.loads(request.config_json)["opencv"]["parameters"]
        except Exception:
            #logging.warning("Invalid or missing parameters in config_json. Using defaults.")
            parameters = {}
        
        try:
            stream_countdown = json.loads(request.config_json)["opencv"].get("stream", None)
        except Exception:
            stream_countdown = None

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
            logging.error(f"Failed to decode input image ({type(e).__name__}): {e}")
            return folder_wd_pb2.Envelope()
            
            
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
                    "runtime": 0.0,
                    "parameters": {"device": "cuda:1"}
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
                return folder_wd_pb2.Envelope()

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
                return folder_wd_pb2.Envelope()

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
            "runtime": time.time() - start_time,
            "parameters": {"device": "cuda:1"},
            #pass the stream_coutndown if any
            "stream": stream_countdown,
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
