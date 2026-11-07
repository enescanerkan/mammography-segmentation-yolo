"""HTTP API and the single-page client of the positioning review app.

    GET    /api/health                              model status, rules, storage usage, client defaults
    GET    /api/studies                             recent studies
    POST   /api/studies                             multipart ``files`` (DICOMs or a study folder) -> study record
    DELETE /api/studies                             delete every stored study
    GET    /api/studies/{id}                        full study record
    GET    /api/studies/{id}/labels                 coordinates only, in the 640 frame
    PUT    /api/studies/{id}/labels                 replace edited labels, recompute every measurement
    PUT    /api/studies/{id}/rules                  move the delta-D threshold, re-decide every verdict
    GET    /api/studies/{id}/report?lang=en|tr      written report
    GET    /api/studies/{id}/image/{view}.png       display render
    DELETE /api/studies/{id}                        delete one study

Inference is synchronous per upload: the model set is resident and four views take seconds, so a job queue would
add moving parts without buying anything. Uploaded bytes never touch the disk (see :mod:`.uploads`).
"""
from __future__ import annotations

import logging
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ... import __version__
from ...inference import ModelLoader
from .config import AppConfig, load_app_config
from .service import Engine, PositioningService, ServiceError
from .uploads import InMemoryMultipartReader, UploadError

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONFIG_ENV = "BREAST_SEG_APP_CONFIG"
log = logging.getLogger("breast_seg.positioning_app")


def create_app(config: Optional[AppConfig] = None, engine: Optional[Engine] = None,
               loader: Optional[ModelLoader] = None, service: Optional[PositioningService] = None,
               load_in_background: bool = True) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: application config (defaults when None).
        engine: an inference engine to use instead of building one from the registry (tests inject a fake).
        loader: model loader for the default engine (``(spec) -> model``); Ultralytics when None.
        service: a fully built service (overrides ``config`` / ``engine`` / ``loader``).
        load_in_background: load the models in a thread at start-up so ``/api/health`` answers immediately.
    """
    config = config or AppConfig()
    service = service or PositioningService.from_config(config, engine=engine, loader=loader)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.start(background=load_in_background)
        yield

    app = FastAPI(title=config.title, version=__version__, lifespan=lifespan)
    app.state.service = service
    app.add_middleware(GZipMiddleware, minimum_size=2048)  # the Tabler stylesheet shrinks about sevenfold

    @app.exception_handler(ServiceError)
    async def service_error(_, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def unhandled(_, exc: Exception):
        log.error("unhandled error: %s", traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": "The server hit an unexpected error."})

    # ── api ──────────────────────────────────────────────────────────────
    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return service.health()

    @app.get("/api/studies")
    def list_studies() -> Dict[str, Any]:
        return service.list_studies()

    @app.post("/api/studies")
    async def create_study(request: Request) -> Dict[str, Any]:
        service.require_ready()  # before reading a body that could be hundreds of megabytes
        try:
            reader = InMemoryMultipartReader(
                request.headers.get("content-type", ""), field="files",
                max_files=config.upload.max_files, max_file_bytes=config.upload.max_file_mb * 1024 * 1024)
            files = await reader.read(request.stream())
        except UploadError as exc:
            raise ServiceError(400, str(exc)) from None
        return await run_in_threadpool(service.ingest, files)

    @app.delete("/api/studies")
    def delete_all_studies() -> Dict[str, Any]:
        return service.delete_all()

    @app.get("/api/studies/{study_id}")
    def get_study(study_id: str) -> Dict[str, Any]:
        return service.study(study_id)

    @app.get("/api/studies/{study_id}/labels")
    def get_labels(study_id: str) -> Dict[str, Any]:
        return service.labels(study_id)

    @app.put("/api/studies/{study_id}/labels")
    def put_labels(study_id: str, payload: Any = Body(...)) -> Dict[str, Any]:
        return service.put_labels(study_id, payload)

    @app.put("/api/studies/{study_id}/rules")
    def put_rules(study_id: str, payload: Any = Body(...)) -> Dict[str, Any]:
        return service.put_rules(study_id, payload)

    @app.get("/api/studies/{study_id}/report")
    def get_report(study_id: str, lang: Optional[str] = None) -> Dict[str, Any]:
        return service.report(study_id, lang)

    @app.get("/api/studies/{study_id}/image/{image_id}.png")
    def get_image(study_id: str, image_id: str):
        return FileResponse(service.image_path(study_id, image_id), media_type="image/png",
                            headers={"Cache-Control": "private, max-age=86400"})

    @app.delete("/api/studies/{study_id}")
    def delete_study(study_id: str) -> Dict[str, Any]:
        return service.delete(study_id)

    # ── client ───────────────────────────────────────────────────────────
    @app.middleware("http")
    async def no_stale_client(request: Request, call_next):
        """Make the browser revalidate the client's own code every time.

        Without this a reload can serve a cached module against freshly changed HTML, which shows up as a function
        that exists on disk and not in the page. Renders keep their long cache — they are immutable once written.
        """
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".js", ".css", ".html", ".svg")):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(STATIC_DIR / "assets" / "icon.svg", media_type="image/svg+xml")

    return app


def serve(config: Optional[AppConfig] = None, host: Optional[str] = None, port: Optional[int] = None,
          loader: Optional[ModelLoader] = None) -> None:
    """Run the app with uvicorn (blocking). ``host`` / ``port`` override ``config.server``."""
    import uvicorn

    config = config or AppConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(config, loader=loader)
    uvicorn.run(app, host=host or config.server.host, port=int(port or config.server.port), log_level="info")


def app_from_env() -> FastAPI:
    """No-argument factory for ASGI servers: reads the config named by ``$BREAST_SEG_APP_CONFIG`` (defaults if unset).

    ``uvicorn --factory breast_seg.apps.positioning.api:app_from_env --host 127.0.0.1 --port 8000``
    """
    path = os.environ.get(CONFIG_ENV, "").strip()
    return create_app(load_app_config(path or None))
