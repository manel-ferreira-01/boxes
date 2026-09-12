"""boxes_client -- a thin client for calling deployed AI "boxes" by IP:port.

A *box* is one of the gRPC services in ``/images/`` built to the shared
``pipeline.PipelineService.Process(Envelope) -> Envelope`` interface. Point
this client at any one by address and send an ``Envelope``::

    from boxes_client import Box
    b = Box("localhost:8061")            # local box   ("10.0.0.5:8061" for remote)

    # any box -- generic, box-agnostic (data + config dicts)
    res = b.run(data={"sentences": ["hello", "world"]},
                config={"my_box": {"command": "do_thing"}})

The core (``Box`` / ``Result`` / envelope builders) knows **no box**.  There is
also an *optional convenience layer* for specific boxes -- e.g. the tapnext
point-tracking one-liner -- kept separate on purpose::

    from boxes_client import trace                      # optional, tapnext-only
    res = trace(b, images=["frame.jpg"], grid_size=30)

No registry, no central server: the client connects directly to the box. Local
and remote boxes are the same call -- just change the address.
"""

import os as _os
import sys as _sys

# Vendored generated proto modules import each other and ``aux`` as top-level
# names; make that directory reachable before they are imported.
_PB_DIR = _os.path.join(_os.path.dirname(__file__), "_pb")
if _os.path.isdir(_PB_DIR) and _PB_DIR not in _sys.path:
    _sys.path.append(_PB_DIR)

from .box import Box
from .result import Result
from .envelope import load
from .conveniences import trace  # optional, convenience layer (not core)

__all__ = ["Box", "Result", "load", "trace"]
__version__ = "0.1.0"
