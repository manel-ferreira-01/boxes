#!/usr/bin/env python3
"""Test LightGlue - uses carro.jpg and eiffel.png (two good test images)"""

import grpc
import sys
sys.path.append("opencv_box/protos")
sys.path.append("opencv_box/test")

from aux import wrap_value, unwrap_value
import pipeline_pb2 as opencv_pb2
import pipeline_pb2_grpc as opencv_pb2_grpc

import numpy as np
import json
import io
import os

def bytes_to_np(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b))


# Load exactly 2 images for matching
img1_path = "/home/manuelf/boxes/opencv_box/test/00.jpg"
img2_path = "/home/manuelf/boxes/opencv_box/test/01.jpg"

print("="*60)
print("LightGlue Test: carro.jpg + eiffel.png")
print("="*60)

with open(img1_path, 'rb') as f: img_bytes_1 = f.read()
with open(img2_path, 'rb') as f: img_bytes_2 = f.read()

img_list = [img_bytes_1, img_bytes_2]
print(f"Loaded 2 images ({len(img_bytes_1)/1024:.1f}KB + {len(img_bytes_2)/1024:.1f}KB)")


# Test configurations  
lg_config = {"opencv": {"parameters": {"feature_extractor": "SUPERPOINT", "max_keypoints": 500}}}
sift_config = {"opencv": {"parameters": {"feature_extractor": "SIFT", "max_keypoints": 1024}}}


channel_opts = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
channel = grpc.insecure_channel('localhost:8061', options=channel_opts)
stub = opencv_pb2_grpc.PipelineServiceStub(channel)

print("\n" + "="*60)  
print("TEST 1: LightGlue (SuperPoint)")
print("="*60)

try:
    response_lg = stub.Process(opencv_pb2.Envelope(
        config_json=json.dumps(lg_config),
        data={"images": wrap_value(img_list)}
    ))
    
    cfg = json.loads(response_lg.config_json)
    print(f"✓ Success! Runtime: {cfg.get('runtime', 0):.3f}s")  
    if 'matcher' in cfg: print(f"  Matcher: {cfg['matcher']}")
    
    for key in response_lg.data.keys():
        try:
            arr = bytes_to_np(unwrap_value(response_lg.data[key]))
            print(f"  {key}: shape={arr.shape}")
        except: pass

except Exception as e:
    print(f"✗ Failed: {type(e).__name__}")
    import traceback; traceback.print_exc()


print("\n" + "="*60)
print("TEST 2: SIFT (backward compatible)")  
print("="*60)

try:
    response_sift = stub.Process(opencv_pb2.Envelope(
        config_json=json.dumps(sift_config),
        data={"images": wrap_value(img_list)}
    ))
    
    cfg = json.loads(response_sift.config_json)
    print(f"✓ Success! Runtime: {cfg.get('runtime', 0):.3f}s")
    
    for key in response_sift.data.keys():
        try:
            arr = bytes_to_np(unwrap_value(response_sift.data[key]))
            print(f"  {key}: shape={arr.shape}")
        except: pass

except Exception as e:
    print(f"✗ Failed: {type(e).__name__}")


print("\nDone!")
