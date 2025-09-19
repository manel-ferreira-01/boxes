import pipeline_pb2

def wrap_value(obj):
    """Wrap a Python object into a pipeline.Value"""
    if isinstance(obj, bool):
        return pipeline_pb2.Value(flag=obj)
    elif isinstance(obj, int):
        return pipeline_pb2.Value(i=obj)
    elif isinstance(obj, float):
        return pipeline_pb2.Value(f=obj)
    elif isinstance(obj, str):
        return pipeline_pb2.Value(s=obj)
    elif isinstance(obj, bytes):
        return pipeline_pb2.Value(b=obj)

    elif isinstance(obj, list):
        if all(isinstance(v, bool) for v in obj):
            return pipeline_pb2.Value(flags=pipeline_pb2.BoolList(values=obj))
        elif all(isinstance(v, int) for v in obj):
            return pipeline_pb2.Value(ii=pipeline_pb2.IntList(values=obj))
        elif all(isinstance(v, float) for v in obj):
            return pipeline_pb2.Value(ff=pipeline_pb2.FloatList(values=obj))
        elif all(isinstance(v, str) for v in obj):
            return pipeline_pb2.Value(ss=pipeline_pb2.StringList(values=obj))
        elif all(isinstance(v, (bytes, bytearray)) for v in obj):
            return pipeline_pb2.Value(bb=pipeline_pb2.BytesList(values=obj))
    raise TypeError(f"Cannot wrap object of type {type(obj)}: {obj}")


def unwrap_value(val: pipeline_pb2.Value):
    """Unwrap a pipeline.Value into a plain Python object"""
    kind = val.WhichOneof("kind")
    if kind == "flag":
        return val.flag
    if kind == "i":
        return val.i
    if kind == "f":
        return val.f
    if kind == "s":
        return val.s
    if kind == "b":
        return val.b
    if kind == "flags":
        return list(val.flags.values)
    if kind == "ii":
        return list(val.ii.values)
    if kind == "ff":
        return list(val.ff.values)
    if kind == "ss":
        return list(val.ss.values)
    if kind == "bb":
        return list(val.bb.values)
    return None
