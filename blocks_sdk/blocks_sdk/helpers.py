"""Helper functions for package-agnostic image conversion and Protobuf Envelope serialization."""
import os
import io
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Union
from blocks_sdk.protos import pipeline_pb2


def to_bytes(obj: Any) -> bytes:
    """
    Package-agnostic converter to convert image objects, file paths, base64 strings,
    or binary objects into raw image bytes. Supports PIL Images, OpenCV/NumPy arrays,
    file paths, base64, and raw bytes without hard library requirements.
    """
    if obj is None:
        return b""

    # 1. Raw bytes or bytearray
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)

    # 2. File path (str or pathlib.Path) or Base64 string
    if isinstance(obj, (str, Path)):
        str_path = str(obj)
        # Existing file on disk
        if os.path.isfile(str_path):
            with open(str_path, "rb") as f:
                return f.read()
        # Data URI base64 (e.g. data:image/jpeg;base64,...)
        if str_path.startswith("data:") and ";base64," in str_path:
            base64_data = str_path.split(";base64,")[1]
            return base64.b64decode(base64_data)
        # Pure base64 string attempt
        try:
            return base64.b64decode(str_path)
        except Exception:
            return str_path.encode("utf-8")

    # 3. PIL Image (duck-typing: object has callable .save method)
    if hasattr(obj, "save") and callable(obj.save):
        buf = io.BytesIO()
        fmt = getattr(obj, "format", None) or "JPEG"
        try:
            obj.save(buf, format=fmt)
        except Exception:
            obj.save(buf, format="PNG")
        return buf.getvalue()

    # 4. OpenCV / NumPy Array (duck-typing: object has .shape and .dtype)
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        # Try OpenCV imencode if available
        try:
            import cv2
            success, encoded_img = cv2.imencode(".jpg", obj)
            if success:
                return encoded_img.tobytes()
        except ImportError:
            pass

        # Try PIL Image.fromarray if PIL is available
        try:
            from PIL import Image
            buf = io.BytesIO()
            pil_img = Image.fromarray(obj)
            pil_img.save(buf, format="JPEG")
            return buf.getvalue()
        except ImportError:
            pass

        # Fallback to buffer export
        if hasattr(obj, "tobytes"):
            return obj.tobytes()

    raise TypeError(f"Cannot convert object of type {type(obj)} to bytes")


def wrap_value(obj: Any) -> pipeline_pb2.Value:
    """Convert Python object (scalars, images, lists) into a pipeline.Value Protobuf message."""
    if hasattr(obj, "DESCRIPTOR") and getattr(obj.DESCRIPTOR, "name", "") == "Value":
        return obj

    # Single Image object or Path -> check if file or image representation
    if isinstance(obj, Path) or (isinstance(obj, str) and os.path.isfile(obj)) or (hasattr(obj, "save") and callable(obj.save)):
        return pipeline_pb2.Value(b=to_bytes(obj))

    # Primitive types
    if isinstance(obj, (bytes, bytearray)):
        return pipeline_pb2.Value(b=bytes(obj))
    elif isinstance(obj, str):
        return pipeline_pb2.Value(s=obj)
    elif isinstance(obj, float):
        return pipeline_pb2.Value(f=obj)
    elif isinstance(obj, int):
        return pipeline_pb2.Value(f=float(obj))
    elif isinstance(obj, bool):
        return pipeline_pb2.Value(s=str(obj))

    # Lists / Sequences
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return pipeline_pb2.Value(ss=pipeline_pb2.StringList(values=[]))

        # Check if list of image objects / file paths / bytes
        if all(
            isinstance(v, (bytes, bytearray, Path)) or
            (isinstance(v, str) and os.path.isfile(v)) or
            (hasattr(v, "save") and callable(v.save)) or
            (hasattr(v, "shape") and hasattr(v, "dtype"))
            for v in obj
        ):
            byte_list = [to_bytes(v) for v in obj]
            return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=byte_list))

        if all(isinstance(v, (bytes, bytearray)) for v in obj):
            return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=[bytes(v) for v in obj]))

        elif all(isinstance(v, str) for v in obj):
            return pipeline_pb2.Value(ss=pipeline_pb2.StringList(values=obj))

        elif all(isinstance(v, (float, int)) for v in obj):
            return pipeline_pb2.Value(ff=pipeline_pb2.FloatList(values=[float(v) for v in obj]))

        else:
            # Fallback: convert everything to string list or bytes list
            try:
                byte_list = [to_bytes(v) for v in obj]
                return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=byte_list))
            except Exception:
                str_list = [str(v) for v in obj]
                return pipeline_pb2.Value(ss=pipeline_pb2.StringList(values=str_list))

    raise TypeError(f"Cannot wrap type {type(obj)} into pipeline.Value")


def unwrap_value(val: Any) -> Any:
    """Convert pipeline.Value or Protobuf field back into a native Python object."""
    if val is None:
        return None

    if hasattr(val, "WhichOneof"):
        kind = val.WhichOneof("kind")
        if kind == "b":
            return val.b
        elif kind == "s":
            return val.s
        elif kind == "f":
            return val.f
        elif kind == "bb":
            return list(val.bb.values)
        elif kind == "ss":
            return list(val.ss.values)
        elif kind == "ff":
            return list(val.ff.values)

    return val


def wrap_envelope(data: Dict[str, Any] = None, config: Any = None) -> pipeline_pb2.Envelope:
    """Construct a pipeline.Envelope Protobuf message from Python dicts/primitives."""
    if data is None:
        data = {}

    if config is None:
        config_json = "{}"
    elif isinstance(config, dict):
        config_json = json.dumps(config)
    elif isinstance(config, str):
        config_json = config
    else:
        config_json = json.dumps(config)

    wrapped_data = {k: wrap_value(v) for k, v in data.items()}

    return pipeline_pb2.Envelope(
        config_json=config_json,
        data=wrapped_data
    )


def unwrap_envelope(envelope: Any) -> Dict[str, Any]:
    """Unwrap a pipeline.Envelope into a clean Python dictionary."""
    if envelope is None:
        return {}

    config_raw = getattr(envelope, "config_json", "{}")
    config = {}
    if config_raw:
        try:
            config = json.loads(config_raw)
        except Exception:
            config = {"raw_config": config_raw}

    data_dict = {}
    if hasattr(envelope, "data"):
        for key, val in envelope.data.items():
            data_dict[key] = unwrap_value(val)

    # Return merged dict with direct key access + config key
    res = dict(data_dict)
    res["_config"] = config
    return res
