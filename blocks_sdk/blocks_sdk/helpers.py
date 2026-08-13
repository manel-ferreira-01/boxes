"""Helper functions for protobuf value conversion."""
from typing import Any
from blocks_sdk.protos import pipeline_pb2


def wrap_value(obj: Any) -> Any:
    """Convert Python object to pipeline.Value protobuf type."""
    if hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "Value":
        return obj

    if isinstance(obj, bytes):
        return pipeline_pb2.Value(b=obj)
    elif isinstance(obj, str):
        return pipeline_pb2.Value(s=obj)
    elif isinstance(obj, float):
        return pipeline_pb2.Value(f=obj)
    elif isinstance(obj, int):
        return pipeline_pb2.Value(f=float(obj))
    elif isinstance(obj, list) and all(isinstance(v, bytes) for v in obj):
        return pipeline_pb2.Value(
            bb=pipeline_pb2.BytesList(values=obj)
        )
    elif isinstance(obj, list) and all(isinstance(v, str) for v in obj):
        return pipeline_pb2.Value(
            ss=pipeline_pb2.StringList(values=obj)
        )
    elif isinstance(obj, list) and all(isinstance(v, (float, int)) for v in obj):
        return pipeline_pb2.Value(
            ff=pipeline_pb2.FloatList(values=[float(v) for v in obj])
        )
    else:
        raise TypeError(f"Cannot wrap type {type(obj)}")


def unwrap_value(val: Any) -> Any:
    """Convert pipeline.Value or dynamic protobuf field to native Python object."""
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
