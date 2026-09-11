"""Best-effort decoding of Envelope data values.

Boxes send heavy payloads as raw ``bytes``. The most common encodings are:
  * ``numpy`` produced via ``np_to_bytes`` (opencv_box pattern) -> plain ndarrays
  * ``torch.save`` (vggt / tapnext pattern)              -> need torch to load
  * plain image bytes (JPEG/PNG)                          -> keep as bytes
  * JSON strings                                          -> parse if valid

This module tries each encoder in turn and falls back to the raw ``bytes``
on any failure. Nothing here raises: the client always returns *something*.
"""

import io
import json

import numpy as np


def _try_numpy(buf: bytes):
    try:
        arr = np.frombuffer(buf, dtype=np.float32)
        return arr
    except Exception:
        pass
    try:
        arr = np.frombuffer(buf, dtype=np.uint8)
        if arr.nbytes == 0:
            return None
        return arr
    except Exception:
        return None


def _try_torch(buf: bytes):
    try:
        import torch  # noqa: W061
    except ImportError:
        return None
    try:
        obj = torch.load(io.BytesIO(buf), weights_only=False, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
    except Exception:
        pass
    return None


def _try_json(buf: bytes):
    try:
        s = buf.decode("utf-8")
    except Exception:
        return None
    if not s or s[0] not in "{[":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def decode_payload(buf):
    """Return the decoded value of ``buf``.

    Order: JSON (if it looks like it) -> torch (tapnext / vggt) -> numpy
    (opencv_box ``np_to_bytes``) -> raw ``bytes``.
    Nothing raises: the client always returns *something*.
    """
    if buf is None:
        return None
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        return buf
    buf = bytes(buf)

    # JSON first: a JSON document is never a valid torch/numpy buffer.
    # _try_json only fires when the first byte is '{' or '[' — cheap and safe.
    j = _try_json(buf)
    if j is not None:
        return j

    # torch next — tapnext / vggt use torch.save.
    t = _try_torch(buf)
    if t is not None:
        return t

    # numpy next — opencv_box uses np_to_bytes; only attempt when the size is
    # at least one element and not clearly a JPEG.
    if len(buf) >= 4 and len(buf) % 4 == 0 and buf[:2] not in (b"\xff\xd8", b"\x89P"):
        arr = _try_numpy(buf)
        if arr is not None:
            return arr

    return buf
