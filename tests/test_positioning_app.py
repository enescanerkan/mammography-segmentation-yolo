"""The review app over HTTP (FastAPI TestClient), with fake models and phantom DICOMs built in memory."""
from __future__ import annotations

import io
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from breast_seg.apps.positioning import AppConfig, ConfigError, app_config_from_dict, create_app, load_app_config  # noqa: E402
from breast_seg.apps.positioning.storage import CountChanged, StudyStore  # noqa: E402
from breast_seg.apps.positioning.uploads import InMemoryMultipartReader, UploadError  # noqa: E402
from breast_seg.imaging.phantom import phantom_dicom  # noqa: E402
from breast_seg.inference import BrightenSearch  # noqa: E402
from breast_seg.inference.testing import fake_loader, placeholder_weights  # noqa: E402
from breast_seg.positioning import CRITERIA_VERSION  # noqa: E402

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
                 "/static/js/dataset.js", "/static/js/api.js", "/static/js/editor.js", "/static/js/overlay.js",
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


def test_pictures_are_refused_with_the_reason_they_cannot_be_measured(client):
    """A PNG / JPG has no pixel spacing and no view or side: the answer says so once, and names the files."""
    response = client.post("/api/studies", files=[("files", ("L-CC.png", b"\x89PNG....", "image/png")),
                                                  ("files", ("R-CC.jpg", b"\xff\xd8....", "image/jpeg"))])
    answer = response.json()
    assert response.status_code == 422 and answer["code"] == "only_dicom"
    assert answer["params"] == {"names": "L-CC.png, R-CC.jpg"}
    assert answer["detail"] == ("This application reads DICOM (.dcm) only; PNG/JPG files have no pixel spacing or view "
                                "information, so mm cannot be measured. Not used: L-CC.png, R-CC.jpg.")
    # a picture among other unusable files: the usual refusal, with the picture's own line
    mixed = client.post("/api/studies", files=[("files", ("a.png", b"x", "image/png")),
                                               ("files", ("b.txt", b"x", "text/plain"))]).json()
    assert mixed["code"] == "nothing_processed"
    assert [entry["code"] for entry in mixed["params"]["files"]] == ["raster_image", "not_an_image"]


def test_upload_notes_are_stored_with_their_codes(client, study):
    """The notes of a stored study keep their codes, so the page and the report write them in the reader's language."""
    assert study["problem_codes"] == [{"text": "case/notes.txt: skipped — a .txt file, not an image",
                                       "code": "not_an_image", "params": {"file": "case/notes.txt", "suffix": "txt"}}]
    assert [entry["text"] for entry in study["problem_codes"]] == study["problems"]
    picture = client.post("/api/studies", files=_files(("files", ("case/L-CC.png", b"x", "image/png")))).json()
    assert picture["problem_codes"][-1]["code"] == "raster_image"
    report = client.get(f"/api/studies/{picture['id']}/report?lang=tr").json()
    notes = [row["text"] for row in report["sections"][-1]["rows"]]
    assert "case/L-CC.png: atlandı — bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve görünüm " \
           "bilgisi olmadığı için mm ölçülemez." in notes


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
        assert response.json()["code"] == "models_missing"


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


def _stale(client, study, version):
    """Rewrite the stored record as if an older (or newer) app had saved it."""
    store = client.app.state.service.store
    record = store.load(study["id"])
    record["criteria"]["version"] = version
    for image in record["images"]:
        image["criteria"] = [{"id": "sharpness", "name": "Focus", "grade": "poor", "value": 1.0, "unit": "",
                              "detail": "", "view_key": image["view_key"], "estimated": False}]
    record["summary"]["marker"] = "as stored"                  # anything a recompute would drop
    store.save(study["id"], record)
    return record


def test_stale_record_regrades_criteria_in_memory_only(client, study):
    stored = _stale(client, study, 1)
    served = client.get(f"/api/studies/{study['id']}").json()
    assert served["criteria"]["version"] == CRITERIA_VERSION
    assert all(entry["id"] != "sharpness" for image in served["images"] for entry in image["criteria"])
    assert served["summary"] == stored["summary"] and served["rules"] == stored["rules"]
    assert [image["measures"] for image in served["images"]] == [image["measures"] for image in stored["images"]]
    assert client.app.state.service.store.load(study["id"]) == stored     # nothing written back by a read


def test_a_record_stored_with_rounded_millimetres_is_served_with_the_unrounded_ones(client, study):
    """Stored by the version before: the summary rounded to two decimals next to the unrounded per-view values. The
    card (summary) and the image footer (per view) then showed one distance as two numbers; served, they are one."""
    store = client.app.state.service.store
    record = store.load(study["id"])
    exact = [(side["landmark"]["pnl_mm"], side["landmark"]["cc_mm"], side["landmark"]["delta_mm"],
              side["comparison"].get("gap_mm")) for side in record["summary"]["sides"]]
    for side in record["summary"]["sides"]:
        for route in ("landmark", "mask"):
            if side[route].get("ok"):
                for key in ("pnl_mm", "cc_mm", "delta_mm"):
                    side[route][key] = round(side[route][key], 2)
        if "gap_mm" in side["comparison"]:
            side["comparison"]["gap_mm"] = round(side["comparison"]["gap_mm"], 2)
    record["criteria"]["version"] = 7
    store.save(study["id"], record)
    assert any(side["landmark"]["pnl_mm"] != pnl for side, (pnl, *_) in zip(record["summary"]["sides"], exact))

    served = client.get(f"/api/studies/{study['id']}").json()
    assert [(side["landmark"]["pnl_mm"], side["landmark"]["cc_mm"], side["landmark"]["delta_mm"],
             side["comparison"].get("gap_mm")) for side in served["summary"]["sides"]] == exact
    views = {image["view_key"]: image["measures"]["landmark"]["mm"] for image in served["images"]}
    assert all(side["landmark"]["cc_mm"] == views[f"{side['laterality']}-CC"] for side in served["summary"]["sides"])
    assert [side["landmark"]["verdict"] for side in served["summary"]["sides"]] == [
        side["landmark"]["verdict"] for side in record["summary"]["sides"]]              # no verdict was touched
    assert store.load(study["id"]) == record                                             # and nothing written back


def test_newer_record_is_served_untouched(client, study):
    stored = _stale(client, study, 99)
    assert client.get(f"/api/studies/{study['id']}").json() == stored


def test_failed_regrade_serves_the_stored_record(client, study, monkeypatch):
    stored = _stale(client, study, 1)

    def broken(record):
        raise RuntimeError("regrade failed")

    monkeypatch.setattr(client.app.state.service.assessor, "regrade", broken)
    assert client.get(f"/api/studies/{study['id']}").json() == stored


def test_a_stale_record_is_regraded_once_until_it_is_rewritten(client, study, monkeypatch):
    service = client.app.state.service
    _stale(client, study, 1)
    calls = []
    regrade = service.assessor.regrade
    monkeypatch.setattr(service.assessor, "regrade", lambda record: calls.append(record["id"]) or regrade(record))
    first = client.get(f"/api/studies/{study['id']}").json()
    assert client.get(f"/api/studies/{study['id']}").json() == first
    assert client.get(f"/api/studies/{study['id']}/report?lang=tr").status_code == 200
    assert calls == [study["id"]]                                  # one regrade for three requests
    # rewritten on disk (as another process would): the cached copy is not served again
    record = _stale(client, study, 2)
    record["summary"]["marker"] = "rewritten"
    service.store.save(study["id"], record)
    assert client.get(f"/api/studies/{study['id']}").json()["summary"]["marker"] == "rewritten"
    assert len(calls) == 2
    # an edit saves the record at the current version: nothing is regraded, nothing stale is served
    edited = client.put(f"/api/studies/{study['id']}/rules", json={"threshold_mm": 12}).json()
    assert client.get(f"/api/studies/{study['id']}").json() == edited and len(calls) == 2
    assert client.delete(f"/api/studies/{study['id']}").status_code == 200
    assert client.get(f"/api/studies/{study['id']}").status_code == 404


def test_a_failed_regrade_is_logged_once(client, study, monkeypatch, caplog):
    stored = _stale(client, study, 1)

    def broken(record):
        raise RuntimeError("regrade failed")

    monkeypatch.setattr(client.app.state.service.assessor, "regrade", broken)
    with caplog.at_level("WARNING", logger="breast_seg.positioning_app"):
        for _ in range(3):
            assert client.get(f"/api/studies/{study['id']}").json() == stored
    assert sum("could not regrade" in message for message in caplog.messages) == 1


def test_unknown_and_malformed_ids(client):
    for path in ("/api/studies/doesnotexist", "/api/studies/..%2Fetc", "/api/studies/abc/labels",
                 "/api/studies/abc/report"):
        assert client.get(path).status_code == 404, path


def test_delete_all(client, study):
    client.post("/api/studies", files=_files())
    assert client.delete("/api/studies").json() == {"deleted": 2}
    assert client.get("/api/studies").json() == {"studies": [], "total": 0, "unopened": 0}


def test_delete_all_refuses_a_count_the_user_did_not_confirm(client, study):
    assert client.get("/api/studies").json()["total"] == 1
    client.post("/api/studies", files=_files())                    # one more arrived after the confirmation
    refused = client.delete("/api/studies?expected=1")
    assert refused.status_code == 409 and "2 studies are stored now, not 1" in refused.json()["detail"]
    assert client.get("/api/studies").json()["total"] == 2         # nothing was deleted
    assert client.delete("/api/studies?expected=2").json() == {"deleted": 2}


# ── refusals the page translates ───────────────────────────────────────────
def test_a_refusal_a_reader_can_meet_carries_its_code_and_parameters(client, study, tmp_path, monkeypatch):
    from breast_seg.apps.positioning.service import ERRORS

    missing = client.get("/api/studies/doesnotexist")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Study not found.", "code": "study_not_found", "params": {}}
    assert client.delete("/api/studies/doesnotexist").json()["code"] == "study_not_found"
    assert client.post("/api/studies/doesnotexist/opened").json()["code"] == "study_not_found"

    refused = client.delete("/api/studies?expected=5").json()          # "clear" confirmed for another count
    assert refused["code"] == "clear_changed" and refused["params"] == {"stored": 1, "expected": 5}
    assert refused["detail"] == ERRORS["clear_changed"].format(stored=1, expected=5)

    empty = client.post("/api/studies", files=[("other", ("x.dcm", b"1", "application/dicom"))]).json()
    assert empty == {"detail": "No files were sent.", "code": "no_files", "params": {}}

    # no file of the upload could be used: every file's reason comes with its own code and parameters
    blind = phantom_dicom("CC", "L", width=300, height=400, include_view=False)
    nothing = client.post("/api/studies", files=[("files", ("notes.txt", b"hello", "text/plain")),
                                                 ("files", ("scan/IM1.dcm", blind, "application/dicom")),
                                                 ("files", ("scan/IM2.dcm", b"not a dicom", "application/dicom"))])
    answer = nothing.json()
    assert nothing.status_code == 422 and answer["code"] == "nothing_processed"
    files = answer["params"]["files"]
    assert [(file["code"], file["params"].get("file")) for file in files] == [
        ("not_an_image", "notes.txt"), ("view_unknown", "scan/IM1.dcm"), ("not_dicom", "scan/IM2.dcm")]
    assert files[0]["params"] == {"file": "notes.txt", "suffix": "txt"} and files[2]["params"]["why"]
    assert answer["detail"].split("\n") == [ERRORS["nothing_processed"], *(file["text"] for file in files)]

    # the service's own reasons: an analysis that fails, and the second file of that view
    def broken(pixels, meta):
        raise RuntimeError("out of memory")
    monkeypatch.setattr(client.app.state.service.engine, "analyse", broken)
    view = phantom_dicom("CC", "L", width=300, height=400)
    failed = client.post("/api/studies", files=[("files", ("a/L-CC.dcm", view, "application/dicom")),
                                                ("files", ("b/L-CC.dcm", view, "application/dicom"))]).json()
    assert [(file["code"], file["params"]) for file in failed["params"]["files"]] == [
        ("inference_failed", {"file": "a/L-CC.dcm", "why": "out of memory"}),
        ("duplicate_view", {"file": "b/L-CC.dcm", "view": "L-CC", "first": "a/L-CC.dcm"})]

    # a refusal only a programmer meets says what is wrong, and has no code
    header = client.post("/api/studies", files=_files(), headers={"X-Study-Key": "short"}).json()
    assert set(header) == {"detail"}


def test_upload_refusals_carry_their_limits(tmp_path):
    config = _config(tmp_path, upload={"max_files": 2, "max_file_mb": 1})
    with TestClient(create_app(config, loader=fake_loader(), load_in_background=False)) as client:
        many = client.post("/api/studies", files=_files()).json()
        assert many == {"detail": "At most 2 files can be uploaded at once.", "code": "too_many_files",
                        "params": {"limit": 2}}
        big = phantom_dicom("CC", "L", width=900, height=1200)  # > 1 MB
        large = client.post("/api/studies", files=[("files", ("L-CC.dcm", big, "application/dicom"))]).json()
        assert large["code"] == "nothing_processed"
        assert large["params"]["files"] == [{"text": "L-CC.dcm: file is too large", "code": "file_too_large",
                                             "params": {"file": "L-CC.dcm"}}]


def _js_table(name):
    """``{key: text}`` of one translation table of i18n.js (single-quoted values, one per line)."""
    source = (REPO / "src/breast_seg/apps/positioning/static/js/i18n.js").read_text(encoding="utf-8")
    block = source[source.index(f"const {name} = {{"):]
    return dict(re.findall(r"^\s*(\w+): '([^']*)',?$", block[:block.index("};")], flags=re.M))


def test_the_page_and_the_server_order_the_views_alike():
    """One grid order, R-MLO, L-MLO, R-CC, L-CC: the page's VIEW_ORDER and breast_seg.positioning.study.VIEW_ORDER."""
    from breast_seg.positioning.study import VIEW_ORDER

    source = (REPO / "src/breast_seg/apps/positioning/static/js/main.js").read_text(encoding="utf-8")
    found = re.search(r"^const VIEW_ORDER = \[([^\]]*)\];$", source, flags=re.M)
    assert found and tuple(re.findall(r"'([^']*)'", found.group(1))) == VIEW_ORDER


def test_every_refusal_code_has_a_translation_with_the_same_placeholders():
    import ast

    from breast_seg.apps.positioning import service
    from breast_seg.imaging import dicom

    placeholders = lambda text: set(re.findall(r"\{(\w+)\}", text))  # noqa: E731
    turkish = _js_table("ERRORS_TR")
    assert set(turkish) == set(service.ERRORS)
    for code, english in service.ERRORS.items():
        assert placeholders(turkish[code]) == placeholders(english), code

    # every reason a file is not used: the reader's are named where they are raised (a code, and the parameters
    # its sentence quotes), the service's are listed in the test above
    raised = {}
    for node in ast.walk(ast.parse(Path(dicom.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "DicomError":
            assert len(node.args) == 2 and isinstance(node.args[1], ast.Constant), ast.dump(node)[:120]
            raised[node.args[1].value] = {keyword.arg for keyword in node.keywords}
    assert set(raised) == set(dicom.ERROR_CODES) and all("file" in names for names in raised.values())
    raised.update(file_too_large={"file"}, duplicate_view={"file", "view", "first"},
                  analysis_refused={"file", "why"}, inference_failed={"file", "why"})
    reasons = _js_table("PROBLEMS_TR")
    assert set(reasons) == set(service.PROBLEMS) == set(raised)
    for code, names in raised.items():
        assert placeholders(reasons[code]) == names, code
    # the report writes a stored study's upload notes by the same codes, with the same Turkish lines
    from breast_seg.positioning.report import PROBLEMS
    assert PROBLEMS["tr"] == reasons


def test_the_recent_list_reads_only_the_rows_it_shows(client, study, monkeypatch):
    """A long archive must not be parsed for the landing page, and counting it (the first "clear" click) reads no
    record at all."""
    import os
    import time as clock

    store = client.app.state.service.store
    older = client.post("/api/studies", files=_files()).json()["id"]
    newest = client.post("/api/studies", files=_files()).json()["id"]
    now = clock.time()
    for age, study_id in enumerate([newest, older, study["id"]]):     # written one minute apart, newest last
        os.utime(store.study_dir / study_id / "study.json", (now - 60 * age, now - 60 * age))
    loaded = []
    load = store.load
    monkeypatch.setattr(store, "load", lambda study_id: loaded.append(study_id) or load(study_id))

    listing = client.get("/api/studies?limit=2").json()
    assert listing["total"] == 3 and [row["id"] for row in listing["studies"]] == [newest, older]
    assert sorted(loaded) == sorted([newest, older])                  # the third record was never opened
    loaded.clear()
    assert client.get("/api/studies?limit=0").json() == {"studies": [], "total": 3, "unopened": 0}
    assert loaded == []
    assert len(client.get("/api/studies").json()["studies"]) == 3     # the default shows them all here
    assert client.get("/api/studies?limit=-5").json()["studies"] == []


KEY = "ab" * 32


def test_a_dataset_study_is_found_again_by_its_key(client, tmp_path):
    """The page sends a salted hash with a dataset study; the same hash finds the stored record (and what was
    corrected in it) when the folder is opened again. Nothing else about the study is stored."""
    health = client.get("/api/health").json()
    salt = health["dataset"]["key_salt"]
    assert len(salt) == 64 and int(salt, 16) >= 0
    assert client.get("/api/health").json()["dataset"]["key_salt"] == salt

    created = client.post("/api/studies", files=_files(), headers={"X-Study-Key": KEY}).json()
    plain = client.post("/api/studies", files=_files()).json()        # a single-study upload carries no key
    other = "cd" * 32
    found = client.post("/api/studies/lookup", json={"keys": [KEY, other, "not a key", 7]}).json()
    assert found == {"found": {KEY: created["id"]}}
    assert KEY not in (tmp_path / "data" / "studies" / created["id"] / "study.json").read_text()
    assert not (tmp_path / "data" / "studies" / plain["id"] / "study.key").exists()

    # a correction made to the stored study is what the lookup leads back to
    before = created["images"][2]  # R-CC
    moved = [before["landmarks"]["points"]["nipple"][0] - 40.0, before["landmarks"]["points"]["nipple"][1]]
    payload = {"images": [{"view_key": "R-CC", "polygons": before["polygons"],
                           "landmarks": {"points": {"nipple": moved}}, "lesions": before["lesions"]}]}
    assert client.put(f"/api/studies/{created['id']}/labels", json=payload).status_code == 200
    again = client.post("/api/studies/lookup", json={"keys": [KEY]}).json()["found"][KEY]
    stored = client.get(f"/api/studies/{again}").json()
    assert again == created["id"] and stored["edited"]
    assert stored["images"][2]["landmarks"]["points"]["nipple"] == moved

    # uploaded twice under one key (two tabs): the lookup names the newer record
    second = client.post("/api/studies", files=_files(), headers={"X-Study-Key": KEY}).json()
    assert client.post("/api/studies/lookup", json={"keys": [KEY]}).json()["found"][KEY] == second["id"]

    # a deleted study is not found, and the key of another installation never matches
    for study_id in (second["id"], created["id"]):
        assert client.delete(f"/api/studies/{study_id}").status_code == 200
    assert client.post("/api/studies/lookup", json={"keys": [KEY]}).json() == {"found": {}}


def _listed(client):
    listing = client.get("/api/studies").json()
    return [row["id"] for row in listing["studies"]], listing["total"], listing["unopened"]


def test_a_study_analysed_ahead_is_not_listed_until_somebody_opens_it(client, tmp_path):
    """A dataset analyses the next study ahead of the reviewer. It is stored, but "Recent studies" is the list of
    studies somebody looked at: the server lists it once a client opens it, by whatever route, or edits it."""
    seen = client.post("/api/studies", files=_files()).json()["id"]                   # a single upload: opened
    ahead = [client.post("/api/studies", files=_files(), headers={"X-Study-Unopened": "1", "X-Study-Key": key}).json()
             for key in (KEY, "cd" * 32, "ef" * 32)]
    assert _listed(client) == ([seen], 4, 3)
    assert client.get("/api/studies?limit=0").json() == {"studies": [], "total": 4, "unopened": 3}
    assert client.get(f"/api/studies/{ahead[0]['id']}").status_code == 200           # reading it is not opening it
    assert _listed(client)[2] == 3

    # opened for review (what the page says when the study is on screen), from any client
    assert client.post(f"/api/studies/{ahead[0]['id']}/opened").json() == {"opened": ahead[0]["id"]}
    assert client.post(f"/api/studies/{ahead[0]['id']}/opened").status_code == 200   # again: nothing changes
    assert sorted(_listed(client)[0]) == sorted([seen, ahead[0]["id"]]) and _listed(client)[2] == 2
    assert client.post("/api/studies/nosuchstudy/opened").status_code == 404

    # an edit is a review too, whoever makes it: corrected labels, or a moved threshold
    view = ahead[1]["images"][2]
    moved = [view["landmarks"]["points"]["nipple"][0] - 20.0, view["landmarks"]["points"]["nipple"][1]]
    client.put(f"/api/studies/{ahead[1]['id']}/labels", json={"images": [{
        "view_key": view["view_key"], "polygons": view["polygons"], "lesions": view["lesions"],
        "landmarks": {"points": {"nipple": moved}}}]})
    client.put(f"/api/studies/{ahead[2]['id']}/rules", json={"threshold_mm": 12})
    assert len(_listed(client)[0]) == 4 and _listed(client)[2] == 0

    # the mark is on disk, next to the record and not in it: a restarted server reads the same list
    fresh = client.post("/api/studies", files=_files(), headers={"X-Study-Unopened": "1"}).json()["id"]
    rows, total, unopened = StudyStore(tmp_path / "data").listing()
    assert fresh not in [row["id"] for row in rows] and (len(rows), total, unopened) == (4, 5, 1)
    assert "unopened" not in (tmp_path / "data" / "studies" / fresh / "study.json").read_text()
    assert client.delete("/api/studies?expected=5").json() == {"deleted": 5}          # "clear" removes them as well
    assert _listed(client) == ([], 0, 0)


def test_an_incomplete_study_is_analysed_again_unless_it_was_edited(client, monkeypatch):
    """A study with a file that failed in a way another attempt can change (here the models failing on one view for
    the moment) must not be what the folder gets back for ever. Its key is not offered to the next lookup; the next
    upload replaces it. Once a reviewer has edited it, it is kept and offered: corrections are never passed over."""
    engine = client.app.state.service.engine
    analyse = engine.analyse
    failing = {"on": True}

    def flaky(pixels, meta):
        if failing["on"] and meta["view_key"] == "L-CC":
            raise RuntimeError("out of memory")
        return analyse(pixels, meta)

    monkeypatch.setattr(engine, "analyse", flaky)

    def find():
        return client.post("/api/studies/lookup", json={"keys": [KEY]}).json()["found"].get(KEY)

    def upload():
        response = client.post("/api/studies", files=_files(), headers={"X-Study-Key": KEY})
        assert response.status_code == 200, response.text
        return response.json()

    first = upload()                                                   # one view failed
    assert [entry["code"] for entry in first["problem_codes"]] == ["inference_failed"] and find() is None
    second = upload()                                                  # the folder is opened again: analysed again
    assert client.get(f"/api/studies/{first['id']}").status_code == 404           # ... in place of the old record
    assert client.get("/api/studies").json()["total"] == 1 and find() is None

    view = second["images"][2]                                         # a reviewer corrects the incomplete study
    moved = [view["landmarks"]["points"]["nipple"][0] - 20.0, view["landmarks"]["points"]["nipple"][1]]
    client.put(f"/api/studies/{second['id']}/labels", json={"images": [{
        "view_key": view["view_key"], "polygons": view["polygons"], "lesions": view["lesions"],
        "landmarks": {"points": {"nipple": moved}}}]})
    assert find() == second["id"]                                      # edited: it is what the folder gets back
    again = upload()                                                   # another incomplete upload does not replace it
    assert client.get(f"/api/studies/{second['id']}").status_code == 200 and find() == second["id"]

    failing["on"] = False
    whole = upload()                                                   # every file used
    assert find() == whole["id"] and not whole["problems"]
    assert client.get(f"/api/studies/{again['id']}").status_code == 404           # the unedited incomplete one went
    assert client.get(f"/api/studies/{second['id']}").status_code == 200          # the edited one is still there

    bad = client.post("/api/studies", files=_files(), headers={"X-Study-Key": KEY, "X-Study-Left-Out": "many"})
    assert bad.status_code == 400 and "left out" in bad.json()["detail"]


BROKEN = ("files", ("case/broken.dcm", b"not a dicom file at all", "application/dicom"))


@pytest.mark.parametrize("extra, headers, code", [
    (BROKEN, {}, "not_dicom"),                                                   # not DICOM: the bytes stay the same
    (("files", ("case/scan.png", b"\x89PNG fake", "image/png")), {}, "raster_image"),
    (("files", ("case/x/IM9.dcm", phantom_dicom("LM", "L", width=450, height=600), "application/dicom")), {},
     "view_unsupported"),
    (None, {"X-Study-Left-Out": "3"}, None),                                     # the client's file limit
])
def test_a_failure_that_would_repeat_does_not_make_a_study_incomplete(client, extra, headers, code):
    """A file that is not DICOM, a picture, a view the app does not measure or the files the client's limit left out
    give the same result on every attempt: such a study is complete as stored and found again by its key, instead of
    being analysed afresh (to the same record) every time its folder is opened."""
    files = _files(extra) if extra else _files()
    study = client.post("/api/studies", files=files, headers={"X-Study-Key": KEY, **headers}).json()
    assert [entry["code"] for entry in study["problem_codes"]] == ([code] if code else [])
    assert client.post("/api/studies/lookup", json={"keys": [KEY]}).json() == {"found": {KEY: study["id"]}}
    again = client.post("/api/studies", files=_files(), headers={"X-Study-Key": KEY}).json()
    assert client.get(f"/api/studies/{study['id']}").status_code == 200           # not replaced by a later upload
    assert client.post("/api/studies/lookup", json={"keys": [KEY]}).json() == {"found": {KEY: again["id"]}}


def test_the_second_file_of_a_view_does_not_make_a_study_incomplete(client):
    """A duplicate view is skipped by design and would be skipped again: such a study is found by its key."""
    data = phantom_dicom("CC", "L", width=450, height=600)
    study = client.post("/api/studies", headers={"X-Study-Key": KEY},
                        files=[("files", ("a/L-CC.dcm", data, "application/dicom")),
                               ("files", ("b/L-CC.dcm", data, "application/dicom"))]).json()
    assert study["problems"] and len(study["images"]) == 1
    assert client.post("/api/studies/lookup", json={"keys": [KEY]}).json() == {"found": {KEY: study["id"]}}


def test_key_lookups_are_answered_from_memory(tmp_path, monkeypatch):
    """The study folders are walked once; after that a lookup, a new key, a delete and the retention purge keep the
    answer right without walking them again."""
    store = StudyStore(tmp_path, retention_days=1, purge_every_s=0)
    now = 2.0e9
    for study_id, key in (("a1", KEY), ("b2", "cd" * 32), ("c3", KEY)):
        store.save(study_id, {"id": study_id, "created": now, "images": []})
        (store.study_dir / study_id / "study.key").write_text(key)       # as a server before this run left them
    walks = []
    directories = store._directories
    monkeypatch.setattr(store, "_directories", lambda: walks.append(1) or directories())

    assert store.find_keys([KEY, "cd" * 32, "ef" * 32]) in ({KEY: "c3", "cd" * 32: "b2"}, {KEY: "a1", "cd" * 32: "b2"})
    assert len(walks) == 1                                              # read once ...
    store.save("d4", {"id": "d4", "created": now, "images": []})
    store.tag("d4", key="ef" * 32)
    assert store.find_keys(["ef" * 32]) == {"ef" * 32: "d4"}             # a key set later is known
    store.tag("e5x", key=KEY)                                           # no such study: nothing is remembered
    newest = store.find_keys([KEY])[KEY]
    assert store.delete(newest)
    other = "a1" if newest == "c3" else "c3"
    assert store.find_keys([KEY]) == {KEY: other}                       # the older record of that key, once it is gone
    assert store.delete(other) and store.find_keys([KEY]) == {}
    assert len(walks) == 1                                              # ... and never again for a lookup

    store.save("old", {"id": "old", "created": 1000.0, "images": []})
    store.tag("old", key="12" * 32)
    assert store.find_keys(["12" * 32]) == {"12" * 32: "old"}
    assert store.purge_expired(now) == 1                                # the purge walks, and tells the index
    walks.clear()
    assert store.find_keys(["12" * 32, "cd" * 32]) == {"cd" * 32: "b2"} and walks == []
    assert store.delete_all() == 2 and store.find_keys(["cd" * 32, "ef" * 32]) == {}


def _record(study_id, created=1.0e9):
    return {"id": study_id, "created": created, "images": []}


def test_the_newest_study_of_a_key_is_the_one_created_last_even_after_a_restart(tmp_path):
    """Which study a key opens follows the records' creation times, not the order the folders happen to be read in,
    so a restart never changes it; a record from before ``created`` was stored falls back to its file's time."""
    store = StudyStore(tmp_path)
    for study_id, created in (("zz", 3.0e9), ("aa", 1.0e9), ("mm", 2.0e9)):
        store.save(study_id, _record(study_id, created), key=KEY)
    assert store.find_keys([KEY]) == {KEY: "zz"}
    assert StudyStore(tmp_path).find_keys([KEY]) == {KEY: "zz"}            # read from disk again
    (store.study_dir / "old").mkdir()                                       # as a server before this left it:
    (store.study_dir / "old" / "study.json").write_text('{"id": "old", "images": []}')   # no creation time
    (store.study_dir / "old" / "study.key").write_text(KEY)
    os.utime(store.study_dir / "old" / "study.json", (4.0e9, 4.0e9))
    again = StudyStore(tmp_path)
    assert again.find_keys([KEY]) == {KEY: "old"}
    assert again.delete("old") and again.find_keys([KEY]) == {KEY: "zz"}
    assert again.delete("zz") and StudyStore(tmp_path).find_keys([KEY]) == {KEY: "mm"}


def test_a_study_opened_now_is_recent_however_long_ago_it_was_analysed(tmp_path):
    """Recent is the later of when the record was written and when the study was last opened: a study a dataset
    analysed ahead long ago and opened now comes first, not after the whole archive."""
    store = StudyStore(tmp_path)
    store.save("old", _record("old"), unopened=True)
    os.utime(store.study_dir / "old" / "study.json", (1.0e9, 1.0e9))
    for i in range(3):
        store.save(f"new{i}", _record(f"new{i}", 1.5e9 + i))
        os.utime(store.study_dir / f"new{i}" / "study.json", (1.5e9 + i, 1.5e9 + i))
    assert [row["id"] for row in store.listing(2)[0]] == ["new2", "new1"]
    assert store.mark_opened("old")
    assert [row["id"] for row in store.listing(2)[0]] == ["old", "new2"]
    assert [row["id"] for row in store.listing(10)[0]] == ["old", "new2", "new1", "new0"]


def test_a_new_study_appears_with_its_tags_or_not_at_all(tmp_path, monkeypatch):
    """The record and its key, "unopened" and "incomplete" marks are written into a hidden folder that is renamed
    into place: no lookup, listing or "clear" ever sees the study without its tags, and a crash leaves only a
    hidden folder, which the purge removes."""
    store = StudyStore(tmp_path)
    seen = []
    write_into = StudyStore._write_into

    def watch(directory, record, renders):
        seen.append((directory.name.startswith(".pending-"), sorted(path.name for path in directory.iterdir()),
                     store.path("b2") is None, store.count()))
        write_into(directory, record, renders)

    monkeypatch.setattr(StudyStore, "_write_into", staticmethod(watch))
    store.save("b2", _record("b2", time.time()), {"R-CC": np.zeros((4, 4), np.uint8)}, key=KEY, unopened=True,
               incomplete=True)
    assert seen == [(True, ["study.incomplete", "study.key", "study.unopened"], True, 0)]
    assert sorted(path.name for path in (store.study_dir / "b2").iterdir()) == [
        "R-CC.png", "study.incomplete", "study.json", "study.key", "study.unopened"]
    assert store.listing()[1:] == (1, 1) and store.find_keys([KEY]) == {}  # unopened and incomplete, from the start

    crashed = store.study_dir / ".pending-c3-000000"
    crashed.mkdir()
    (crashed / "study.json").write_text("{}")
    assert store.count() == 1 and store.listing()[1] == 1                 # a half-written study is not a study
    os.utime(crashed, (1.0e9, 1.0e9))
    store.purge_expired()
    assert not crashed.exists() and store.path("b2") is not None


def test_clear_counts_and_deletes_in_one_step(tmp_path, monkeypatch):
    """A "clear" confirmed for n studies deletes nothing when another number is stored, and a study still being
    written is neither counted nor deleted: it is stored once its writing ends."""
    store = StudyStore(tmp_path)
    store.save("a1", _record("a1"))
    with pytest.raises(CountChanged) as caught:
        store.delete_all(expected=2)
    assert caught.value.stored == 1 and store.count() == 1

    writing, done = threading.Event(), threading.Event()
    write_into = StudyStore._write_into

    def slow(directory, record, renders):
        if directory.name.startswith(".pending-"):
            writing.set()
            done.wait(5)
        write_into(directory, record, renders)

    monkeypatch.setattr(StudyStore, "_write_into", staticmethod(slow))
    saver = threading.Thread(target=store.save, args=("b2", _record("b2")))
    saver.start()
    assert writing.wait(5)
    assert store.delete_all(expected=1) == 1                                # only the stored one
    done.set()
    saver.join(5)
    assert store.count() == 1 and store.load("b2") == _record("b2")


def test_an_edit_and_a_replacement_never_cross(tmp_path, monkeypatch):
    """Whether a stale study of a key was edited is read at the moment it is deleted, under the store's write lock,
    and an edit is read, changed and written under the same lock: an edit saved first keeps its study, and an edit
    that comes while the study is being deleted finds it gone instead of leaving a record in a deleted folder."""
    store = StudyStore(tmp_path)
    for study_id, created in (("a1", 1.0e9), ("b2", 1.1e9), ("d4", 1.2e9)):
        store.save(study_id, _record(study_id, created), key=KEY, incomplete=True)
    store.save("c3", _record("c3", 2.0e9), key=KEY)
    assert store.update("a1", lambda record: record.update(edited=True))["edited"]

    checking, result = threading.Event(), {}
    edited = store._edited

    def slow_check(study_id):
        checking.set()
        time.sleep(0.2)                                                     # the edit below arrives meanwhile
        return edited(study_id)

    def edit():
        checking.wait(5)
        result["record"] = store.update("d4", lambda record: record.update(edited=True))

    monkeypatch.setattr(store, "_edited", slow_check)
    editor = threading.Thread(target=edit)
    editor.start()
    removed = store.delete_replaceable(KEY, keep="c3")
    editor.join(5)
    assert removed == ["b2", "d4"] and result == {"record": None}         # the edit waited, then found it gone
    assert not (store.study_dir / "d4").exists() and not (store.study_dir / "b2").exists()
    assert store.load("a1")["edited"] and store.find_keys([KEY]) == {KEY: "c3"}


def test_study_key_validation(client):
    for bad in ("abc", "AB" * 32, "zz" * 32, "ab" * 33):
        response = client.post("/api/studies", files=_files(), headers={"X-Study-Key": bad})
        assert response.status_code == 400 and "study key" in response.json()["detail"].lower()
    assert client.get("/api/studies").json()["total"] == 0            # refused before anything was stored
    for payload in ([KEY], {"keys": KEY}, {}, "text"):
        assert client.post("/api/studies/lookup", json=payload).status_code == 400
    assert client.post("/api/studies/lookup", json={"keys": []}).json() == {"found": {}}


def test_the_key_salt_belongs_to_the_installation(tmp_path):
    """Random per data directory, kept across restarts, never inside a study folder, and not removed by "clear"."""
    first = StudyStore(tmp_path / "one")
    salt = first.salt()
    assert StudyStore(tmp_path / "one").salt() == salt                # a restart keeps it
    assert StudyStore(tmp_path / "two").salt() != salt                # another installation has its own
    assert (tmp_path / "one" / "study_key_salt").read_text() == salt
    assert first.delete_all() == 0 and StudyStore(tmp_path / "one").salt() == salt
    (tmp_path / "one" / "study_key_salt").write_text("damaged")       # unusable: replaced, not served
    replaced = StudyStore(tmp_path / "one").salt()
    assert replaced != salt and len(replaced) == 64


def test_the_recent_list_names_the_views_in_the_grid_order(client):
    """R-MLO, L-MLO, R-CC, L-CC, whatever order the files came in."""
    files = list(reversed(_files()))
    stored = client.post("/api/studies", files=files).json()
    assert [image["view_key"] for image in stored["images"]] == ["L-CC", "R-CC", "L-MLO", "R-MLO"]   # upload order
    row = next(row for row in client.get("/api/studies").json()["studies"] if row["id"] == stored["id"])
    assert row["views"] == ["R-MLO", "L-MLO", "R-CC", "L-CC"]


def test_without_a_nipple_the_segmentation_route_is_estimated_at_the_contour(tmp_path):
    """No nipple found by any search: the segmentation route measures at the farthest tissue contour point (marked
    estimated) and the breast gets its verdict; ``nipple.contour: false`` in the app config turns that off."""
    def upload(**nipple):
        app = create_app(_config(tmp_path, nipple=nipple), loader=fake_loader(nipple_conf=0.001),
                         load_in_background=False)
        with TestClient(app) as client:
            return client.post("/api/studies", files=_files()).json(), client.app.state.service

    record, service = upload()
    assert service.assessor.contour_fallback
    for image in record["images"]:
        assert "nipple" not in image["polygons"]                      # missed at 0.25, by the rescue and brightened
        mask = image["measures"]["mask"]
        assert mask["ok"] and mask["estimated"] and mask["nipple"] == "contour" and mask["missing"] == "no nipple mask"
    for side in record["summary"]["sides"]:
        assert side["mask"]["ok"] and sorted(side["mask"]["estimated"]) == ["CC", "MLO"]

    record, service = upload(contour=False)
    assert not service.assessor.contour_fallback
    assert all(image["measures"]["mask"] == {"ok": False, "reason": "no nipple mask", **(
        {"line": image["measures"]["mask"]["line"]} if "line" in image["measures"]["mask"] else {})}
        for image in record["images"])


def test_the_editor_can_have_a_view_cleaned_without_anything_being_stored(client, study):
    """POST /clean: the contours the editor holds come back cleaned; a contour the rules leave alone comes back as it
    was sent, and the stored record does not change."""
    url = f"/api/studies/{study['id']}/clean"
    view = study["images"][1]                                         # L-MLO
    stored = client.get(f"/api/studies/{study['id']}").json()
    untouched = client.post(url, json={"view_key": view["view_key"], "polygons": view["polygons"]}).json()
    assert untouched["view_key"] == "L-MLO" and set(untouched["polygons"]) <= set(view["polygons"])
    breast = view["polygons"]["breast-tissue"]
    spiked = [list(point) for point in breast["points"]]
    spiked.insert(1, [spiked[0][0] + 1.0, max(spiked[0][1] - 60.0, 0.0)])   # a thin spike off the contour
    island = {"points": [[600, 600], [615, 600], [615, 615], [600, 615]], "confidence": 0.5, "rescued": False}
    sent = {**view["polygons"], "breast-tissue": {**breast, "points": spiked}, "nipple": island}
    reply = client.post(url, json={"view_key": "L-MLO", "polygons": sent}).json()
    cleaned = reply["polygons"]
    assert cleaned["breast-tissue"]["points"] != spiked                # the spike is smoothed away
    assert cleaned.get("nipple", {}).get("points") != island["points"]  # the square nipple is smoothed (or dropped)
    assert {"breast-tissue", "nipple"} <= set(reply["changed"]) <= {"breast-tissue", "pectoral", "nipple"}
    assert client.get(f"/api/studies/{study['id']}").json() == stored    # nothing was stored

    # a second click on what the first returned changes nothing, and neither does it once that has been saved
    again = client.post(url, json={"view_key": "L-MLO", "polygons": cleaned}).json()
    assert again["changed"] == [] and again["polygons"] == cleaned
    edited = {name: {**entry, "edited": True} for name, entry in cleaned.items()}
    saved = client.put(f"/api/studies/{study['id']}/labels", json={"images": [{
        "view_key": "L-MLO", "polygons": edited, "lesions": view["lesions"],
        "landmarks": view["landmarks"]}]}).json()
    reopened = next(image for image in saved["images"] if image["view_key"] == "L-MLO")["polygons"]
    assert all(entry.get("cleaned") for entry in reopened.values())
    assert client.post(url, json={"view_key": "L-MLO", "polygons": reopened}).json()["changed"] == []
    assert client.post(url, json={"view_key": "X-CC", "polygons": {}}).status_code == 404
    assert client.post(url, json={"view_key": "L-MLO", "polygons": [1]}).status_code == 400
    assert client.post("/api/studies/missing/clean", json={"view_key": "L-MLO"}).status_code == 404


def test_recent_rows_say_why_a_study_needs_a_look():
    from breast_seg.apps.positioning.storage import attention
    side = lambda **c: {"comparison": c}  # noqa: E731
    assert attention({"sides": [side(state="agree", wide_gap=False)]}) is None
    assert attention({"sides": [side(state="agree", wide_gap=True), side(state="partial")]}) == "wide"
    assert attention({"sides": [side(state="agree", wide_gap=True), side(state="disagree")]}) == "disagree"
    assert attention({"sides": [side(state="partial")]}) == "partial"
    assert attention({"sides": [side(state="none")]}) is None and attention({}) is None


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


def test_a_contour_saved_unchanged_keeps_a_cleaned_mark_only_when_it_names_its_points(client, study):
    """The mark "Smooth / clean labels" leaves is kept on a contour sent back unchanged (the stored contour stays,
    with the mark), and only when it is the fingerprint of that contour's points."""
    from breast_seg.positioning.labels import fingerprint

    view = study["images"][1]                                         # L-MLO
    pectoral = view["polygons"]["pectoral"]
    nipple = view["landmarks"]["points"]["nipple"]

    def saved(mark, shift):
        payload = {"images": [{"view_key": "L-MLO", "lesions": view["lesions"],
                               "polygons": {**view["polygons"], "pectoral": {**pectoral, "cleaned": mark}},
                               "landmarks": {"points": {"nipple": [nipple[0] - shift, nipple[1]]}}}]}
        record = client.put(f"/api/studies/{study['id']}/labels", json=payload).json()
        return next(image for image in record["images"] if image["view_key"] == "L-MLO")["polygons"]["pectoral"]

    assert saved("0123456789abcdef", 20.0) == pectoral                # not the fingerprint of these points: dropped
    kept = saved(fingerprint(pectoral["points"]), 10.0)
    assert kept == {**pectoral, "cleaned": fingerprint(pectoral["points"])}   # the stored contour, with its mark


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


def test_the_app_turns_its_own_nipple_steps_on_for_its_engine_only(tmp_path):
    """The brightened re-search is the app's choice: its engine is built with it (``nipple.brighten``, on by
    default), and a config can turn it off or size its window; the shared registry carries no such switch."""
    def engine(**nipple):
        return create_app(_config(tmp_path, nipple=nipple), loader=fake_loader(),
                          load_in_background=False).state.service.engine

    assert engine().brighten == BrightenSearch(window_mm=50.0, max_gap_mm=None)
    assert engine(window_mm=40, max_gap_mm=3).brighten == BrightenSearch(window_mm=40.0, max_gap_mm=3.0)
    assert engine(brighten=False).brighten is None
    assert "brighten:" not in MODELS_YAML.read_text(encoding="utf-8")


def test_example_config(tmp_path):
    config = load_app_config(REPO / "configs" / "app.example.yaml")
    assert config.nipple == AppConfig().nipple                # the example shows the defaults
    assert config.registry_path.resolve() == MODELS_YAML.resolve()
    assert config.data_dir.resolve() == (REPO / "data").resolve()
    assert config.weights_root is None and config.models.device == "cpu" and config.language == "en"
    assert not hasattr(config, "criteria")          # no criterion has a configurable band
    overridden = load_app_config(REPO / "configs" / "app.example.yaml",
                                 ["models.device=0", "server.port=9001",
                                  "criteria.pectoral_below_nipple_mm={good: 5, acceptable: 0}",
                                  "criteria.sharpness={good: 1, acceptable: 0}"])
    assert overridden.models.device == "0" and overridden.server.port == 9001
    # ... and a config that still carries the retired bands, including the pectoral depth, loads without error
    retired = load_app_config(None, ["criteria.pectoral_angle_deg={good: [1, 2], acceptable: [0, 3]}"])
    assert retired.language == "en"
    old = app_config_from_dict({"criteria": {"sharpness": {"good": 1, "acceptable": 0},
                                             "pectoral_below_nipple_mm": {"good": 0, "acceptable": -10}}})
    assert old.rules.threshold_mm == 10.0


@pytest.mark.parametrize("data,message", [
    ({"language": "de"}, "language"),
    ({"server": {"portt": 1}}, "unknown keys"),
    ({"colour": 1}, "unknown key"),
    ({"criteria": {"made_up": {"good": 1, "acceptable": 2}}}, "unknown keys"),
    ({"rules": {"threshold_mm": 0}}, "threshold_mm"),
    ({"display": {"preview_size": 10}}, "preview_size"),
    ({"nipple": {"window_mm": 0}}, "window_mm"),
    ({"nipple": {"max_gap_mm": -1}}, "max_gap_mm"),
    ({"nipple": {"rescue": True}}, "unknown keys"),
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
