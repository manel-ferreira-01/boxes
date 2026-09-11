"""Result object returned from ``Box`` calls."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict

from .decode_util import decode_payload


def _is_numeric_array(v):
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(v, np.ndarray)


def _decode_any(buf):
    """Decode a single Envelope Value payload.

    - list of byte payloads: decode each element
    - list of strings/floats: pass through
    - bytes: try decode, else return as-is
    - scalars (str/float/int): pass through
    """
    if isinstance(buf, (list, tuple)):
        items = list(buf)
        if items and all(isinstance(x, (bytes, bytearray, memoryview)) for x in items):
            return [decode_payload(x) for x in items]
        return items
    if isinstance(buf, (bytes, bytearray, memoryview)):
        return decode_payload(buf)
    return buf


@dataclass
class Result:
    """A decoded response from a box.

    ``fields`` maps every Envelope data field to its best-effort decoded value
    (see :mod:`boxes_client.decode_util`). Use ``raw`` for the undecoded Envelope.
    Field values can also be read directly, e.g. ``res.tracks``.
    """

    raw: Any = None
    config: Any = None
    fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_envelope(cls, envelope):
        config = envelope.config_json
        if config:
            try:
                config = json.loads(config)
            except (ValueError, TypeError):
                pass

        fields = {}
        for name, val in envelope.data.items():
            kind = val.WhichOneof("kind") if hasattr(val, "WhichOneof") else None
            if kind == "b":
                buf = val.b
            elif kind == "bb":
                buf = list(val.bb.values)
            elif kind == "ss":
                buf = list(val.ss.values)
            elif kind == "ff":
                buf = list(val.ff.values)
            elif kind == "f":
                buf = val.f
            elif kind == "s":
                buf = val.s
            else:
                buf = val
            fields[name] = _decode_any(buf)
        return cls(raw=envelope, config=config, fields=fields)

    def __getattr__(self, name):
        if name.startswith("_") or name in ("raw", "config", "fields"):
            raise AttributeError(name)
        d = self.__dict__.get("fields") or {}
        if name in d:
            return d[name]
        raise AttributeError(
            f"Result has no field {name!r}; available: {sorted(d)}"
        )

    def as_dict(self):
        return {
            "config": self.config,
            "fields": {
                k: (v.tolist() if _is_numeric_array(v) else v)
                for k, v in self.fields.items()
            },
        }

    def __repr__(self):
        cfg = self.config if isinstance(self.config, dict) else "<str>"
        return f"<Result fields=[{', '.join(sorted(self.fields))}] config={cfg}>"
