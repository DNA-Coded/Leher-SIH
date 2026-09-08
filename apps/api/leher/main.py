from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apps.api.leher.config import API_TITLE, API_VERSION
from apps.api.leher.routers import ml, catalog
from apps.api.leher.jobs.rakshak_job import execute_rakshak_job
from apps.api.leher.paths import REPO_ROOT

import sys
_jal_chakra_dir = str(REPO_ROOT / "services" / "jal-chakra")
if _jal_chakra_dir not in sys.path:
    sys.path.insert(0, _jal_chakra_dir)
from core.catalog import init_catalog

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DuckDB Catalog
    fabric_path = REPO_ROOT / "fabric"
    app.state.catalog_db = init_catalog(fabric_path)
    
    scheduler.add_job(
        execute_rakshak_job,
        'interval',
        hours=1,
        id="rakshak_hourly",
        replace_existing=True
    )
    scheduler.start()
    yield
    scheduler.shutdown()
    app.state.catalog_db.close()

app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
app.state.scheduler = scheduler

app.include_router(ml.router, prefix="/api/v1/ml")
app.include_router(catalog.router, prefix="/api/v1/catalog")

from apps.api.leher.routers import slices
app.include_router(slices.router, prefix="/api/v1/model")

@app.get("/health")
def health():
    return {"status": "ok"}
