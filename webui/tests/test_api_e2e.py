"""End-to-end: HTTP API -> core -> boxes_client -> fake box (real gRPC) ->
serialized JSON + artifacts.  This is the contract the SPA will rely on.
"""

import io
import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from webui.config import AppEnv
from webui.app import create_app

BOXES_DIR = pathlib.Path(__file__).resolve().parents[1] / "boxes"


@pytest.fixture
def client(tmp_path):
    env = AppEnv(
        host="127.0.0.1", port=0,
        data_dir=tmp_path / "data", boxes_dir=BOXES_DIR,
        artifact_ttl=600, max_artifact_bytes=10_000_000,
        max_upload_bytes=1_000_000,
    )
    app = create_app(env)
    with TestClient(app) as c:
        yield c


def _seed(client, fake, name, def_id):
    r = client.post("/api/fleet", json={
        "name": name, "addr": fake.addr, "def_id": def_id})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------------------------- root

def test_root_lists_defs(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "lang_sam" in body["defs"] and "vggt" in body["defs"]


def test_defs_endpoint_shape(client):
    r = client.get("/api/defs")
    assert r.status_code == 200
    body = r.json()
    assert len(body["defs"]) == 5      # yologpt/opencv out of scope (pre-contract)
    assert "image_upload" in body["vocabulary"]["widgets"]
    assert "overlay" in body["vocabulary"]["visualizers"]


# ----------------------------------------------------------- the happy path

def test_lang_sam_full_round_trip(client, fake_lang_sam):
    fid = _seed(client, fake_lang_sam, "lang sam", "lang_sam")

    # 1) upload a fake JPEG
    up = client.post("/api/upload",
                     files={"file": ("dog.jpg", io.BytesIO(b"jpeg-bytes"),
                                     "image/jpeg")})
    assert up.status_code == 201, up.text
    ref = up.json()["ref"]

    # 2) call the box through the API
    r = client.post("/api/call", json={
        "fleet_id": fid,
        "data": {"images": [ref]},
        "section": {"text_prompt": ["a dog"]},
        "parameters": {"box_threshold": 0.4},
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["box"] == "lang_sam"
    assert body["status"] == "done"
    assert body["error"] is None
    assert body["declared_encoding"] == "zstd_pickle"
    item = body["fields"]["results"][0]
    assert item["bboxes"] == [[1, 2, 30, 40], [5, 6, 7, 8]]
    assert item["scores"] == pytest.approx([0.91, 0.33])
    # bool mask -> inline array of Python bools
    mask = item["masks"][0]
    assert mask["kind"] == "array" and mask["dtype"] == "bool"
    assert all(isinstance(x, bool) for row in mask["values"] for x in row)
    # upload artifact fetchable back byte-identical
    tok = ref.lstrip("@")
    g = client.get(f"/api/file/{tok}")
    assert g.status_code == 200 and g.content == b"jpeg-bytes"


def test_vggt_full_round_trip(client, fake_vggt):
    """The full panel path: namespaced call, torch-decoded tensors with their
    full shape, the GLB as a type-sniffed file artifact."""
    fid = _seed(client, fake_vggt, "vggt lab", "vggt")

    up = [client.post("/api/upload",
                      files={"file": (f"frame{n}.jpg", io.BytesIO(b"jpeg-n"),
                                      "image/jpeg")}) for n in (0, 1)]
    assert all(u.status_code == 201 for u in up), [u.text for u in up]
    refs = [u.json()["ref"] for u in up]

    r = client.post("/api/call", json={
        "fleet_id": fid,
        "data": {"images": refs},
        "parameters": {"conf_threshold": 25},
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["box"] == "vggt" and body["status"] == "done"
    assert body["declared_encoding"]["glb_file"] == "identity"
    assert body["fields"]["world_points"]["shape"] == [17, 3]
    # large tensor -> typed buffer artifact (not an inline array, not pickle)
    depth = body["fields"]["depth"]
    assert depth["kind"] == "buffer" and depth["dtype"] == "float32"
    assert depth["shape"] == [2, 256, 256] and depth["size"] == 256 * 256 * 2 * 4
    tok_d = depth["url"].rsplit("/", 1)[-1]
    raw = client.get(f"/api/file/{tok_d}").content
    assert len(raw) == depth["size"]

    glb = body["fields"]["glb_file"]
    assert glb["kind"] == "file" and glb["mime"] == "model/gltf-binary"
    tok = glb["url"].rsplit("/", 1)[-1]
    g = client.get(f"/api/file/{tok}")
    assert g.status_code == 200 and g.content[:4] == b"glTF"


def test_box_error_surfaces_as_status(client, std_pb):
    """A box answering status=error is a *successful* call that reports an
    error in-band — the client never raises; the UI must not either."""
    import json as _json
    from conftest import FakeBox
    pb2, pb2_grpc, aux = std_pb

    class FakeBroken(pb2_grpc.PipelineServiceServicer):
        def Process(self, request, context):
            return pb2.Envelope(config_json=_json.dumps(
                {"lang_sam": {"status": "error",
                              "error": "model exploded (test)"}}))

    fake = FakeBox(FakeBroken(), pb2, pb2_grpc)
    try:
        fid = _seed(client, fake, "broken", "lang_sam")
        up = client.post("/api/upload",
                         files={"file": ("x.jpg", io.BytesIO(b"img"),
                                         "image/jpeg")})
        r = client.post("/api/call", json={
            "fleet_id": fid,
            "data": {"images": [up.json()["ref"]]},
            "section": {"text_prompt": ["a dog"]},
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "error"
        assert "exploded" in body["error"]
    finally:
        fake.stop()


def test_call_build_error_is_400_with_detail(client, fake_lang_sam):
    fid = _seed(client, fake_lang_sam, "x", "lang_sam")
    up = client.post("/api/upload",
                     files={"file": ("i.jpg", io.BytesIO(b"img"), "image/jpeg")})
    r = client.post("/api/call", json={
        "fleet_id": fid,
        "data": {"images": [up.json()["ref"]]},
        "parameters": {"nope": 1},
    })
    assert r.status_code == 400
    err = r.json()["error"]
    assert "nope" in err["message"]
    assert "box_threshold" in err["known"]


def test_unknown_fleet_entry(client):
    r = client.post("/api/call", json={"fleet_id": "ghost"})
    assert r.status_code == 404


def test_dead_box_is_502(client):
    # entry pointing at a closed port (nothing listens)
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    r = client.post("/api/fleet", json={
        "name": "dead", "addr": f"127.0.0.1:{dead_port}", "def_id": "clip"})
    fid = r.json()["id"]
    r = client.post("/api/call", json={
        "fleet_id": fid, "data": {"texts": ["hello"]}, "timeout": 2})
    assert r.status_code == 502
    assert r.json()["error"]["stage"] == "grpc"


# ------------------------------------------------------------------- fleet

def test_fleet_crud(client):
    e = client.post("/api/fleet", json={
        "name": "clip", "addr": "10.0.0.5:9061", "def_id": "clip"}).json()
    assert e["id"] == "clip"
    assert client.get(f"/api/fleet/{e['id']}").status_code == 200
    u = client.patch(f"/api/fleet/{e['id']}", json={"note": "lab box"})
    assert u.json()["note"] == "lab box"
    assert client.delete(f"/api/fleet/{e['id']}").status_code == 204
    assert client.get(f"/api/fleet/{e['id']}").status_code == 404


def test_fleet_bad_def_rejected(client):
    r = client.post("/api/fleet", json={
        "name": "x", "addr": "h:9061", "def_id": "nope"})
    assert r.status_code == 400


def test_probe_reachable(client, fake_lang_sam):
    fid = _seed(client, fake_lang_sam, "probe", "lang_sam")
    r = client.post(f"/api/fleet/{fid}/probe", json={},
                    params={"timeout": 3})
    assert r.status_code == 200, r.text
    probe = r.json()["last_probe"]
    assert probe["reachable"] is True
    assert probe["reflection"] is True


def test_probe_unreachable(client):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    fid = client.post("/api/fleet", json={
        "name": "dead", "addr": f"127.0.0.1:{dead_port}"}).json()["id"]
    r = client.post(f"/api/fleet/{fid}/probe", params={"timeout": 1})
    assert r.status_code == 200, r.text
    assert r.json()["last_probe"]["reachable"] is False


def test_upload_too_large(client):
    big = b"x" * 1_100_000   # max_upload_bytes == 1_000_000 in fixture
    r = client.post("/api/upload",
                    files={"file": ("big.bin", io.BytesIO(big),
                                    "application/octet-stream")})
    assert r.status_code == 413


def test_artifact_404(client):
    assert client.get("/api/file/upl_nope").status_code == 404
