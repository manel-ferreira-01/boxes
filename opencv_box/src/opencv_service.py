import concurrent.futures as futures
import grpc
import grpc_reflection.v1alpha.reflection as grpc_reflection
import logging
import os
import time
import json
import sys
import io
import cv2
import numpy as np
import time

# Proto imports
sys.path.append("./protos")
import pipeline_pb2 as folder_wd_pb2
import pipeline_pb2_grpc as folder_wd_pb2_grpc
from aux import wrap_value, unwrap_value

_PORT_DEFAULT = 8061
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT_ENV_VAR = 'PORT'

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

    def Process(self, request, context):
        """Process request: extract features and optionally match."""

        #start timer
        start_time = time.time()

        imgs_in = []
        for image_bytes in unwrap_value(request.data["images"]):
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                imgs_in.append(img)

        parameters = json.loads(request.config_json)["opencv"]["parameters"]
        feature_extractor = parameters.get("feature_extractor", "SIFT")
        ratio_thresh = parameters.get("ratio_thresh", 0.75)
        max_keypoints = parameters.get("max_keypoints", 500)

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
