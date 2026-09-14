"""Shared test fixtures: path setup + fake boxes (same pattern as
``boxes_client/tests/fake_box_smoke.py``)."""

import concurrent.futures as futures
import json
import pickle
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WEBUI_SRC = REPO / "webui" / "src"
if str(WEBUI_SRC) not in sys.path:
    sys.path.insert(0, str(WEBUI_SRC))

_REFLECTION = "grpc.reflection.v1alpha.ServerReflection"

#: every FakeBox ever created this process; the autouse fixture stops them
_LIVE: list["FakeBox"] = []


class FakeBox:
    """Run one servicer on a thread at an ephemeral port; autouse cleanup.

    """

    def __init__(self, servicer, pb2, pb2_grpc):
        import grpc
        import grpc_reflection.v1alpha.reflection as grpc_reflection
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        pb2_grpc.add_PipelineServiceServicer_to_server(servicer, self._server)
        names = (
            pb2.DESCRIPTOR.services_by_name["PipelineService"].full_name,
            _REFLECTION,
        )
        grpc_reflection.enable_server_reflection(names, self._server)
        self._port = self._server.add_insecure_port("127.0.0.1:0")
        self._server.start()
        _LIVE.append(self)

    @property
    def addr(self) -> str:
        return f"127.0.0.1:{self._port}"

    def stop(self):
        self._server.stop(None)


@pytest.fixture
def std_pb():
    """Shared-envelope pb modules (the client's vendored gencode)."""
    from boxes_client._pb_loader import get as get_pb
    return get_pb()


@pytest.fixture
def fake_lang_sam(std_pb):
    """Mimics lang_segm: images + text_prompt in; per-item dicts
    (masks/bboxes/scores) zstd_pickle'd into data.results, declared encoding
    in the response config (the real box's contract)."""
    import numpy as np
    import zstandard as zstd
    pb2, pb2_grpc, aux = std_pb

    class FakeLangSegm(pb2_grpc.PipelineServiceServicer):
        def Process(self, request, context):
            cfg = json.loads(request.config_json) if request.config_json else {}
            sc = cfg.get("lang_sam", {})
            if sc.get("command") == "reset":
                return pb2.Envelope(config_json=json.dumps(
                    {"lang_sam": {"status": "done", "action": "reset"}}))
            imgs = aux.unwrap_value(request.data.get("images"))
            if not imgs:
                return pb2.Envelope(config_json=json.dumps(
                    {"lang_sam": {"status": "empty_request"}}))
            prompt = sc.get("text_prompt") or []
            if not prompt:
                return pb2.Envelope(config_json=json.dumps(
                    {"lang_sam": {"status": "error", "error": "no prompt"}}))
            list_img = list(imgs)
            out = []
            for n in range(len(list_img)):
                h, w = 64 + n * 8, 80
                out.append({
                    "masks": [np.ones((h, w), dtype=bool),
                              np.zeros((h, w), dtype=bool)],
                    "bboxes": [[1, 2, 30 + n, 40], [5, 6, 7, 8]],
                    "scores": [0.91 - 0.01 * n, 0.33],
                    "labels": list(prompt),
                })
            blob = zstd.ZstdCompressor().compress(pickle.dumps(out, protocol=4))
            env = pb2.Envelope(config_json=json.dumps(
                {"lang_sam": {"status": "done", "num_images": len(list_img),
                              "encoding": "zstd_pickle"}}))
            env.data["results"].CopyFrom(aux.wrap_value(bytes(blob)))
            return env

    return FakeBox(FakeLangSegm(), pb2, pb2_grpc)


@pytest.fixture(autouse=True)
def _cleanup_boxes():
    """Stop FakeBoxes created during this test (best-effort)."""
    yield
    while _LIVE:
        _LIVE.pop().stop()
