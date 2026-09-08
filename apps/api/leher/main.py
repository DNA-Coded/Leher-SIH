from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apps.api.leher.config import API_TITLE, API_VERSION
from apps.api.leher.routers import ml
from apps.api.leher.jobs.rakshak_job import execute_rakshak_job

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
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

app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
app.state.scheduler = scheduler

app.include_router(ml.router, prefix="/api/v1/ml")

from apps.api.leher.routers import slices
app.include_router(slices.router, prefix="/api/v1/model")

@app.get("/health")
def health():
    return {"status": "ok"}
