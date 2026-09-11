"""``Box`` -- a thin client for one deployed box (a box = a gRPC AI service).

Give it an address (``host:port``) and it can send an ``Envelope`` and get a
decoded ``Result`` back via the shared ``pipeline.PipelineService`` interface.

Local and remote boxes are identical: ``Box("localhost:8061")`` vs
``Box("10.0.0.5:8061")``. No registry, no central server -- the client dials
the box directly (boxes are push-style servers), which preserves the
distributed nature of the fleet.

Two call levels

--------------
:func:`Box.run`    -- the generic workhorse: ``run(data=..., config=..., method="Process")``.
                      No assumption about field names or payload types.

:func:`Box.trace`  -- convenience for the tapnext *image* box:
                      ``trace(images=[...], grid_size=30)`` is sugar over
                      ``run(data={"images": [...]}, config={"tapnext": {
                      "command":"track", "parameters":{"grid_size":30}}},
                     method="Process")``.
"""

from typing import Any, Dict, List, Optional, Union

import grpc

from ._pb_loader import get as _get_pb
from . import envelope as _env
from .result import Result

_SERVICE = "pipeline.PipelineService"
_DEFAULT_PORT = 8061


class Box:
    """A client for a single box.

    Parameters
    ----------
    address:
        ``"host:port"`` or just ``"host"`` (defaults to port 8061).
    port:
        Optional explicit port (alternative to the ``":"`` in the address).
    timeout:
        Per-RPC timeout in seconds (default 600).
    config_key:
        Top-level key used by the :meth:`reset` / :meth:`trace` conveniences
        (default ``"tapnext"``). For :meth:`run` this is irrelevant -- you pass
        the full ``config`` dict yourself.
    """

    def __init__(
        self,
        address: str,
        port: Optional[int] = None,
        timeout: float = 600,
        config_key: str = "tapnext",
    ):
        self.host, self.port = _split_address(address, port)
        self.timeout = timeout
        self.config_key = config_key

        self._channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )
        pb2, pb2_grpc, _aux = _get_pb()
        self._stub = pb2_grpc.PipelineServiceStub(self._channel)

    # ------------------------------------------------------------------ utils
    def close(self):
        try:
            self._channel.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------- discover
    def info(self, timeout: float = 10) -> Dict[str, Any]:
        """Ask the box (via gRPC reflection) what it exposes.

        Returns ``{"service", "methods", "reflection", "reachable"}``.
        ``reflection`` is ``False`` if the box does not serve reflection (the
        client still works; it just cannot self-describe).
        """
        out = {"service": _SERVICE, "methods": [], "reflection": False,
               "reachable": False}
        try:
            grpc.channel_ready_future(self._channel).result(timeout=timeout)
            out["reachable"] = True
        except Exception:
            return out

        try:
            from grpc_reflection.v1alpha import (
                reflection_pb2,
                reflection_pb2_grpc,
            )
        except ImportError:
            return out

        stub = reflection_pb2_grpc.ServerReflectionStub(self._channel)
        try:
            # Bidirectional-streaming RPC: one request -> (take the first) response.
            def _reqs():
                yield reflection_pb2.ServerReflectionRequest(list_services="*")
            resp = next(stub.ServerReflectionInfo(iter(_reqs()), timeout=timeout))
            services = [e.name for e in resp.list_services_response.service]
            if _SERVICE in services:
                out["reflection"] = True
                out["methods"] = ["Process"]
            else:
                out["methods"] = services
        except Exception:
            out["reflection"] = False
        return out

    # ---------------------------------------------------------------- internal
    def _send(self, envelope, method: str) -> Result:
        """Low-level: send ``envelope`` via the named ``method`` on the box."""
        fn = getattr(self._stub, method, None)
        if not callable(fn):
            known = sorted(n for n in dir(self._stub) if not n.startswith("_"))
            raise AttributeError(
                f"Box has no RPC method {method!r}. "
                f"This stub defines: {known} (only 'Process' is guaranteed; "
                f"other methods depend on the box's own .proto)."
            )
        resp = fn(envelope, timeout=self.timeout)
        return Result.from_envelope(resp)

    # ------------------------------------------------------------------ calls
    def run(
        self,
        data: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        method: str = "Process",
        reset_first: bool = False,
    ) -> Result:
        """Generic entry point.

        Parameters
        ----------
        data:
            Mapping of ``field_name -> value`` written into ``Envelope.data``.
            Values follow :mod:`boxes_client.envelope` coercion rules:
            ``bytes`` / ``pathlib.Path`` -> bytes; ``str`` -> literal string;
            ``int`` -> float; lists of those -> ``BytesList`` / ``StringList``
            / ``FloatList``.
        config:
            The box-specific control payload (dict) serialized to
            ``Envelope.config_json``. Shape depends on the box.
        method:
            RPC name to invoke (default ``"Process"``). Boxes may expose extra
            methods (e.g. opencv's ``similarity_check``, yolo's
            ``DetectSequence`` / ``TrackSequence`` / ``AllProcessing``).
        reset_first:
            If ``True``, send a tapnext-style reset (config-only
            ``{config_key: {"command": "reset"}}``) on ``Process`` first.
            Useful for stateful boxes; ignored by stateless ones.
        """
        if reset_first:
            self.reset()
        return self._send(_env.build(data, config), method)

    def reset(self) -> Result:
        """Best-effort clear of server-side state.

        Sends a config-only ``Process`` call with ``{config_key: {"command":
        "reset"}}``. tapnext-style boxes treat this as a hard state reset;
        boxes that don't recognize the command will typically ignore it.
        """
        return self._send(_env.reset_envelope(self.config_key), "Process")

    def trace(
        self,
        images: Union[str, bytes, "pathlib.PurePath", List],
        grid_size: Optional[int] = None,
        reset_first: bool = True,
        **params: Any,
    ) -> Result:
        """Convenience for an image-box (tapnext).

        Builds ``data={"images": [bytes,...]}`` and
        ``config={"tapnext": {"command": "track", "parameters": {...}}}``, then
        :meth:`run`s it via ``Process``. ``images`` may be a single local
        path / preencoded bytes, or a list of either.
        """
        images_bytes = _env._load_images(images)
        if not images_bytes:
            raise ValueError("trace(): no images provided")

        parameters: Dict[str, Any] = dict(params or {})
        if grid_size is not None:
            parameters["grid_size"] = int(grid_size)

        return self.run(
            data={"images": images_bytes},
            config={self.config_key: {"command": "track",
                                       "parameters": parameters}},
            method="Process",
            reset_first=reset_first,
        )

    # Back-compat alias: low-level "I already built the Envelope" call.
    def call(self, envelope: Any) -> Result:
        return self._send(envelope, "Process")


def _split_address(address: str, port: Optional[int]) -> tuple:
    address = (address or "").strip()
    if ":" not in address:
        return (address or "localhost"), (int(port) if port is not None else _DEFAULT_PORT)
    host, _, maybe_port = address.rpartition(":")
    if not host:
        host = "localhost"
    if maybe_port.isdigit():
        return host, int(maybe_port)
    if port is not None:
        return host, int(port)
    return host, _DEFAULT_PORT
