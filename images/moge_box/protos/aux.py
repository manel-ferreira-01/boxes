"""
Helper functions for converting Python objects to/from pipeline.Value oneof.
Canonical copy lives at protos/aux.py in the repo root.
"""

import pipeline_pb2


def wrap_value(obj):
    """Wrap a Python object into a pipeline.Value"""
    if isinstance(obj, float):
        return pipeline_pb2.Value(f=obj)
    elif isinstance(obj, str):
        return pipeline_pb2.Value(s=obj)
    elif isinstance(obj, bytes):
        return pipeline_pb2.Value(b=obj)
    elif isinstance(obj, int):
        return pipeline_pb2.Value(f=float(obj))
    
    elif isinstance(obj, list):
        if len(obj) == 0:
            return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=[]))
        elif all(isinstance(v, (float, int)) for v in obj):
            return pipeline_pb2.Value(ff=pipeline_pb2.FloatList(values=[float(v) for v in obj]))
        elif all(isinstance(v, str) for v in obj):
            return pipeline_pb2.Value(ss=pipeline_pb2.StringList(values=obj))
        elif all(isinstance(v, (bytes, bytearray)) for v in obj):
            return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=list(obj)))
    
    raise TypeError(f"Cannot wrap object of type {type(obj)}: {obj}")


def unwrap_value(val):
    """Unwrap a pipeline.Value into a plain Python object"""
    if val is None:
        return None
    
    kind = val.WhichOneof("kind")
    if kind == "f":
        return val.f
    if kind == "s":
        return val.s
    if kind == "b":
        return val.b
    if kind == "ff":
        return list(val.ff.values)
    if kind == "ss":
        return list(val.ss.values)
    if kind == "bb":
        return list(val.bb.values)
    return None
