"""The review app over HTTP (FastAPI TestClient), with fake models and phantom DICOMs built in memory."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from breast_seg.apps.positioning import AppConfig, ConfigError, app_config_from_dict, create_app, load_app_config  # noqa: E402
from breast_seg.apps.positioning.storage import StudyStore  # noqa: E402
from breast_seg.apps.positioning.uploads import InMemoryMultipartReader, UploadError  # noqa: E402
from breast_seg.imaging.phantom import phantom_dicom  # noqa: E402
from breast_seg.inference.testing import fake_loader, placeholder_weights  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO / "configs" / "models.yaml"
VIEWS = [("R", "MLO"), ("L", "MLO"), ("R", "CC"), ("L", "CC")]


def _config(tmp_path, **sections):
    data = {
        "models": {"registry": str(MODELS_YAML), "weights_root": str(placeholder_weights(MODELS_YAML, tmp_path / "w"))},
        "storage": {"data_dir": str(tmp_path / "data")},
        "display": {"preview_size": 800},
    }
    for key, value in sections.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) else value
    return app_config_from_dict(data, base_dir=tmp_path)


@pytest.fixture()
def client(tmp_path):
    app = create_app(_config(tmp_path), loader=fake_loader(), load_in_background=False)
    with TestClient(app) as test_client:
        yield test_client


def _files(*extra):
    files = [("files", (f"case/{side}{view}/IM{i}.dcm", phantom_dicom(view, side, width=450, height=600, seed=i,
                                                                      include_view=False, include_laterality=False),
                        "application/dicom"))
             for i, (side, view) in enumerate(VIEWS)]
    return files + list(extra)


@pytest.fixture()
def study(client):
    response = client.post("/api/studies", files=_files(("files", ("case/notes.txt", b"hello", "text/plain"))))
    assert response.status_code == 200, response.text
    return response.json()


# ── health and client ──────────────────────────────────────────────────────
def test_health(client):
    health = client.get("/api/health").json()
    assert health["ready"] and not health["loading"] and health["language"] == "en"
    assert health["title"] == "Mammography Positioning Review" and health["device"] == "cpu"
    assert health["rules"] == {"threshold_mm": 10.0, "severe_mm": 15.0}
    assert health["roles"]["segmentation"] == "seg_v4" and health["storage"]["studies"] == 0
    assert all(not model["weights"].startswith("/") for model in health["models"])


def test_client_is_served(client):
    index = client.get("/")
    assert index.status_code == 200 and "<title>Mammography Positioning Review</title>" in index.text
    assert index.headers["cache-control"].startswith("no-store")
    for path in ("/static/app.css", "/static/theme.css", "/static/js/main.js", "/static/js/i18n.js",
                 "/static/vendor/tabler/tabler.min.css", "/static/assets/icon.svg"):
        response = client.get(path)
        assert response.status_code == 200, path
    assert client.get("/favicon.ico").headers["content-type"].startswith("image/svg")
    assert client.head("/").status_code == 200


# ── upload ─────────────────────────────────────────────────────────────────
def test_upload_builds_a_study(study):
    assert [image["view_key"] for image in study["images"]] == ["R-MLO", "L-MLO", "R-CC", "L-CC"]
    assert study["problems"] == ["case/notes.txt: skipped — a .txt file, not an image"]
    image = study["images"][0]
    assert image["view_source"] == "folder name" and image["laterality_source"] == "folder name"
    assert set(image["measures"]) == {"landmark", "mask"} and image["measures"]["landmark"]["ok"]
    assert image["landmarks"]["detected"] and image["polygons"]["breast-tissue"]["points"]
    assert [side["laterality"] for side in study["summary"]["sides"]] == ["L", "R"]
    assert study["criteria"]["summary"]["graded"] > 0
    assert study["models"]["seg_v4"] == "breast_seg_v4_yolo26m.pt" and study["preview_size"] == 800
    assert "masks" not in image and "filename" not in image


def test_uploaded_dicoms_never_reach_the_disk(client, tmp_path, monkeypatch):
    import starlette.formparsers

    async def refuse(*_args, **_kwargs):
        raise AssertionError("Starlette's form parser (which spools to disk) must not be used")

    monkeypatch.setattr(starlette.formparsers.MultiPartParser, "parse", refuse)
    study = client.post("/api/studies", files=_files()).json()
    stored = sorted(p.name for p in (tmp_path / "data" / "studies" / study["id"]).iterdir())
    assert stored == ["L-CC.png", "L-MLO.png", "R-CC.png", "R-MLO.png", "study.json"]
    for path in (tmp_path / "data").rglob("*"):
        if path.is_file():
            assert path.read_bytes()[128:132] != b"DICM"


def test_duplicate_views_are_skipped(client):
    data = phantom_dicom("CC", "L", width=450, height=600)
    study = client.post("/api/studies", files=[("files", ("a/L-CC.dcm", data, "application/dicom")),
                                               ("files", ("b/L-CC.dcm", data, "application/dicom"))]).json()
    assert len(study["images"]) == 1
    assert study["problems"] == ["b/L-CC.dcm: L-CC was already uploaded as a/L-CC.dcm — skipped"]


def test_nothing_usable_is_a_422(client):
    response = client.post("/api/studies", files=[("files", ("x.txt", b"hello", "text/plain"))])
    assert response.status_code == 422 and "No image could be processed" in response.json()["detail"]


def test_bad_uploads(client):
    assert client.post("/api/studies", json={"files": []}).status_code == 400
    response = client.post("/api/studies", files=[("other", ("x.dcm", b"1", "application/dicom"))])
    assert response.status_code == 400 and response.json()["detail"] == "No files were sent."


def test_upload_limits(tmp_path):
    config = _config(tmp_path, upload={"max_files": 2, "max_file_mb": 1})
    with TestClient(create_app(config, loader=fake_loader(), load_in_background=False)) as client:
        assert client.post("/api/studies", files=_files()).status_code == 400
        big = phantom_dicom("CC", "L", width=900, height=1200)  # > 1 MB
        small = phantom_dicom("CC", "R", width=300, height=400)
        study = client.post("/api/studies", files=[("files", ("L-CC.dcm", big, "application/dicom")),
                                                   ("files", ("R-CC.dcm", small, "application/dicom"))]).json()
        assert study["problems"] == ["L-CC.dcm: file is too large"] and len(study["images"]) == 1


def test_not_ready_without_weights(tmp_path):
    config = app_config_from_dict({"models": {"registry": str(MODELS_YAML), "weights_root": str(tmp_path / "none")},
                                   "storage": {"data_dir": str(tmp_path / "data")}})
    with TestClient(create_app(config, loader=fake_loader(), load_in_background=False)) as client:
        health = client.get("/api/health").json()
        assert not health["ready"] and "weights file not found" in health["errors"]["pose_mlo"]
        response = client.post("/api/studies", files=_files())
        assert response.status_code == 503 and "not loaded" in response.json()["detail"]


def test_injected_engine_is_used(tmp_path):
    class Silent:
        ready = False

        def load(self):
            return self

        def status(self):
            return {"ready": False, "loaded": [], "errors": {"pose_mlo": "absent"}}

        def loaded_weights(self):
            return {}

        def analyse(self, pixels, meta):  # pragma: no cover - never reached
            raise AssertionError

    with TestClient(create_app(_config(tmp_path), engine=Silent(), load_in_background=False)) as client:
        assert client.get("/api/health").json()["errors"] == {"pose_mlo": "absent"}
        assert client.post("/api/studies", files=_files()).status_code == 503


# ── studies ────────────────────────────────────────────────────────────────
def test_list_get_image_delete(client, study):
    rows = client.get("/api/studies").json()["studies"]
    assert rows[0]["id"] == study["id"] and rows[0]["views"] == ["R-MLO", "L-MLO", "R-CC", "L-CC"]
    assert client.get(f"/api/studies/{study['id']}").json() == study
    image = client.get(f"/api/studies/{study['id']}/image/R-MLO.png")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
    import cv2

    decoded = cv2.imdecode(np.frombuffer(image.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (800, 800)
    assert client.get(f"/api/studies/{study['id']}/image/X-MLO.png").status_code == 404
    assert client.delete(f"/api/studies/{study['id']}").json() == {"deleted": study["id"]}
    assert client.get(f"/api/studies/{study['id']}").status_code == 404
    assert client.delete(f"/api/studies/{study['id']}").status_code == 404


def test_unknown_and_malformed_ids(client):
    for path in ("/api/studies/doesnotexist", "/api/studies/..%2Fetc", "/api/studies/abc/labels",
                 "/api/studies/abc/report"):
        assert client.get(path).status_code == 404, path


def test_delete_all(client, study):
    client.post("/api/studies", files=_files())
    assert client.delete("/api/studies").json() == {"deleted": 2}
    assert client.get("/api/studies").json() == {"studies": []}


def test_labels_round_trip(client, study):
    labels = client.get(f"/api/studies/{study['id']}/labels").json()
    assert labels["frame"] == {"width": 640, "height": 640} and len(labels["images"]) == 4
    before = study["images"][2]  # R-CC
    nipple = before["landmarks"]["points"]["nipple"]
    moved = [nipple[0] - 40.0, nipple[1]]
    payload = {"images": [{"view_key": "R-CC", "polygons": before["polygons"],
                           "landmarks": {"points": {"nipple": moved}}, "lesions": before["lesions"]}]}
    updated = client.put(f"/api/studies/{study['id']}/labels", json=payload).json()
    after = updated["images"][2]
    assert after["landmarks"]["edited"] and after["landmarks"]["points"]["nipple"] == moved
    assert after["measures"]["landmark"]["mm"] != before["measures"]["landmark"]["mm"]
    assert updated["edited"] and not updated["images"][0]["landmarks"].get("edited")
    assert client.get(f"/api/studies/{study['id']}").json()["images"][2]["landmarks"]["points"]["nipple"] == moved


def test_labels_validation(client, study):
    url = f"/api/studies/{study['id']}/labels"
    assert client.put(url, json={"images": [{"view_key": "X-CC"}]}).status_code == 400
    assert client.put(url, json={"images": [{"view_key": "R-CC", "polygons": [1, 2]}]}).status_code == 400
    assert client.put(url, json={"nope": 1}).status_code == 400
    assert client.put("/api/studies/missing/labels", json={"images": [{"view_key": "R-CC"}]}).status_code == 404


def test_rules(client, study):
    url = f"/api/studies/{study['id']}/rules"
    strict = client.put(url, json={"threshold_mm": 1}).json()
    assert strict["rules"] == {"threshold_mm": 1.0, "severe_mm": 15.0}
    deltas = [side["landmark"]["delta_mm"] for side in strict["summary"]["sides"] if side["landmark"]["ok"]]
    verdicts = [side["landmark"]["verdict"] for side in strict["summary"]["sides"] if side["landmark"]["ok"]]
    assert verdicts == ["bad" if d > 1 else "good" for d in deltas]
    assert client.put(url, json={"threshold_mm": 0}).status_code == 400
    assert client.put(url, json={"threshold_mm": "x"}).status_code == 400
    assert client.put(url, json={"threshold_mm": 30}).json()["rules"]["severe_mm"] == 30.0


def test_report(client, study):
    url = f"/api/studies/{study['id']}/report"
    default = client.get(url).json()
    assert default["lang"] == "en" and default["title"] == "Mammography Positioning Report"
    turkish = client.get(url, params={"lang": "tr"}).json()
    assert turkish["lang"] == "tr" and turkish["title"] == "Mamografi Pozisyonlama Raporu"
    assert turkish["text"].startswith("MAMOGRAFİ POZİSYONLAMA RAPORU")  # Turkish dotted capital İ
    assert client.get(url, params={"lang": "de"}).json()["lang"] == "en"


def test_turkish_default_language(tmp_path):
    config = _config(tmp_path, language="tr")
    with TestClient(create_app(config, loader=fake_loader(), load_in_background=False)) as client:
        assert client.get("/api/health").json()["language"] == "tr"
        study = client.post("/api/studies", files=_files()).json()
        assert client.get(f"/api/studies/{study['id']}/report").json()["lang"] == "tr"


def test_background_loading_reports_progress(tmp_path):
    with TestClient(create_app(_config(tmp_path), loader=fake_loader())) as client:
        client.app.state.service.wait_until_loaded(30)
        health = client.get("/api/health").json()
        assert health["ready"] and not health["loading"]


# ── storage, uploads, config ───────────────────────────────────────────────
def test_retention_purge(tmp_path):
    store = StudyStore(tmp_path, retention_days=1, purge_every_s=0)
    store.save("old", {"id": "old", "created": 1000.0, "images": []})
    store.save("new", {"id": "new", "created": 2.0e9, "images": []})
    assert store.purge_if_due() == 1 and [row["id"] for row in store.recent()] == ["new"]
    assert StudyStore(tmp_path, retention_days=0).purge_expired() == 0
    assert store.path("../new") is None and store.image_file("new", "../x") is None
    with pytest.raises(ValueError):
        store.save("../x", {})


def test_multipart_reader_in_small_chunks():
    boundary = "XyZ"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"note\"\r\n\r\nhello\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"a/RCC/1.dcm\"\r\n"
            f"Content-Type: application/dicom\r\n\r\n").encode() + bytes(range(256)) * 3 + \
        (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"big.dcm\"\r\n\r\n").encode() + \
        b"x" * 50 + f"\r\n--{boundary}--\r\n".encode()
    reader = InMemoryMultipartReader(f"multipart/form-data; boundary={boundary}", max_files=5, max_file_bytes=40)
    stream = io.BytesIO(body)
    while chunk := stream.read(7):
        reader.feed(chunk)
    files = reader.finish()
    assert [f.filename for f in files] == ["a/RCC/1.dcm", "big.dcm"]
    assert files[0].too_large and files[0].data == b""  # 768 bytes > 40
    assert files[1].too_large
    with pytest.raises(UploadError):
        InMemoryMultipartReader("application/json")
    limited = InMemoryMultipartReader(f"multipart/form-data; boundary={boundary}", max_files=1)
    with pytest.raises(UploadError, match="At most 1"):
        limited.feed(body)


def test_example_config(tmp_path):
    config = load_app_config(REPO / "configs" / "app.example.yaml")
    assert config.registry_path.resolve() == MODELS_YAML.resolve()
    assert config.data_dir.resolve() == (REPO / "data").resolve()
    assert config.weights_root is None and config.models.device == "cpu" and config.language == "en"
    assert config.criteria["sharpness"] == {"good": 820, "acceptable": 330}
    overridden = load_app_config(REPO / "configs" / "app.example.yaml",
                                 ["models.device=0", "server.port=9001", "criteria.sharpness={good: 1, acceptable: 0}"])
    assert overridden.models.device == "0" and overridden.server.port == 9001
    assert overridden.criteria["sharpness"] == {"good": 1, "acceptable": 0}
    assert overridden.criteria["symmetry_mm"] == {"good": 10, "acceptable": 20}


@pytest.mark.parametrize("data,message", [
    ({"language": "de"}, "language"),
    ({"server": {"portt": 1}}, "unknown keys"),
    ({"colour": 1}, "unknown key"),
    ({"criteria": {"made_up": {"good": 1, "acceptable": 2}}}, "unknown keys"),
    ({"rules": {"threshold_mm": 0}}, "threshold_mm"),
    ({"display": {"preview_size": 10}}, "preview_size"),
])
def test_invalid_config(data, message):
    with pytest.raises(ConfigError, match=message):
        app_config_from_dict(data)


def test_default_config_is_valid():
    config = AppConfig()
    assert config.server.port == 8000 and config.display.preview_size == 1280 and config.rules.to_rules().threshold_mm == 10


def test_app_from_env(tmp_path, monkeypatch):
    import yaml

    from breast_seg.apps.positioning import api

    config = {"models": {"registry": str(MODELS_YAML), "weights_root": str(tmp_path / "none")},
              "storage": {"data_dir": "data"}, "language": "tr"}
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv(api.CONFIG_ENV, str(path))
    with TestClient(api.app_from_env()) as client:
        assert client.get("/api/health").json()["language"] == "tr"
    assert (tmp_path / "data" / "studies").is_dir()  # relative to the config file
