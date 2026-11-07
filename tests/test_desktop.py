"""Desktop launcher: config resolution, the synthetic self-test study, the local server."""
import json
import socket

import pytest

from breast_seg.apps import desktop


def test_desktop_config_uses_the_bundle_and_the_user_folder(tmp_path):
    bundle, home = tmp_path / "bundle", tmp_path / "home"
    home.mkdir()
    cfg = desktop.desktop_config(bundle, home)
    assert cfg.registry_path == bundle / "configs" / "models.yaml"
    assert cfg.weights_root == bundle / "weights"
    assert cfg.data_dir == home / "data"
    assert cfg.models.device == "cpu" and cfg.server.host == "127.0.0.1"


def test_user_yaml_overrides_single_keys(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "app.yaml").write_text("language: tr\nrules: {threshold_mm: 12}\nmodels: {device: '0'}\n", encoding="utf-8")
    cfg = desktop.desktop_config(tmp_path / "bundle", home)
    assert cfg.language == "tr" and cfg.rules.threshold_mm == 12 and cfg.models.device == "0"
    assert cfg.weights_root == tmp_path / "bundle" / "weights"  # untouched keys keep the desktop default


def test_free_port_falls_back_when_the_preferred_port_is_taken():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        taken = busy.getsockname()[1]
        port = desktop.free_port(taken)
    assert port != taken and port > 0


@pytest.mark.parametrize("jpeg2000", [False, True])
def test_synthetic_dicom_is_a_readable_mammogram(jpeg2000):
    pytest.importorskip("pydicom")
    if jpeg2000:
        pytest.importorskip("openjpeg")
    from breast_seg.imaging import read_dicom

    pixels, meta = read_dicom(desktop.synthetic_dicom("CC", "R", jpeg2000=jpeg2000), "R-CC.dcm")
    assert pixels.shape == (2400, 1900)
    assert meta["view"] == "CC" and meta["laterality"] == "R"


def test_local_server_serves_the_app(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/api/health")
    def health():
        return {"ready": True}

    server = desktop.LocalServer(app, desktop.free_port(0)).start()
    try:
        assert server.health() == {"ready": True}
        assert json.loads(server.get("/api/health"))["ready"] is True
    finally:
        server.stop()
    assert not server.thread.is_alive()
