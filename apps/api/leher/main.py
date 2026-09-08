import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apps.api.leher.config import (
    API_TITLE, API_VERSION, ALLOWED_ORIGINS, CACHE_TTL_SECONDS, ENV
)
from apps.api.leher.routers import ml, catalog
from apps.api.leher.jobs.rakshak_job import execute_rakshak_job
from apps.api.leher.paths import REPO_ROOT

import sys
_jal_chakra_dir = str(REPO_ROOT / "services" / "jal-chakra")
if _jal_chakra_dir not in sys.path:
    sys.path.insert(0, _jal_chakra_dir)
from core.catalog import init_catalog

logger = logging.getLogger("leher.api")

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DuckDB Catalog
    fabric_path = REPO_ROOT / "fabric"
    app.state.catalog_db = init_catalog(fabric_path)

    # Schedule Rakshak hourly — does not execute immediately
    scheduler.add_job(
        execute_rakshak_job,
        'interval',
        hours=1,
        id="rakshak_hourly",
        replace_existing=True
    )
    scheduler.start()
    logger.info("Leher API started (env=%s)", ENV)
    yield
    # Shutdown: clean teardown
    scheduler.shutdown(wait=False)
    app.state.catalog_db.close()
    logger.info("Leher API shutdown complete")

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    docs_url="/api/docs",
    lifespan=lifespan
)
app.state.scheduler = scheduler

# CORS — production-appropriate configuration
# No wildcard when credentials are enabled; configurable via LEHER_ALLOWED_ORIGINS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "X-Dataset-Id",
        "X-Selected-Variable",
        "X-Selected-Time",
        "X-Selected-Depth",
        "X-Slice-Rows",
        "X-Slice-Cols",
        "X-Data-Min",
        "X-Data-Max",
        "X-Variables",
    ],
)

# ---------- Global exception handler ----------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unhandled exceptions.
    Returns a safe JSON error without exposing filesystem internals.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# ---------- Routers ----------
app.include_router(ml.router, prefix="/api/v1/ml")
app.include_router(catalog.router, prefix="/api/v1/catalog")

from apps.api.leher.routers import slices, vectors
app.include_router(slices.router, prefix="/api/v1/model")
app.include_router(vectors.router, prefix="/api/v1/vectors")

@app.get("/health")
def health():
    """Liveness probe — always returns 200 if the process is up."""
    return {"status": "ok"}

@app.get("/api/v1/status")
def status(request: Request):
    """Operational readiness endpoint. Reports scheduler + catalog state."""
    from apps.api.leher.jobs.rakshak_job import get_execution_status

    catalog_ok = hasattr(request.app.state, "catalog_db") and request.app.state.catalog_db is not None
    scheduler_running = scheduler.running if scheduler else False

    return {
        "status": "ok" if catalog_ok else "degraded",
        "catalog_initialized": catalog_ok,
        "scheduler_running": scheduler_running,
        "rakshak": get_execution_status(),
        "environment": ENV,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }
