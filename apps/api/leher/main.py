from fastapi import FastAPI
from apps.api.leher.config import API_TITLE, API_VERSION
from apps.api.leher.routers import ml

app = FastAPI(title=API_TITLE, version=API_VERSION)

app.include_router(ml.router, prefix="/api/v1/ml")

@app.get("/health")
def health():
    return {"status": "ok"}
