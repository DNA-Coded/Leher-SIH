import json
import polars as pl
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from apps.api.leher.paths import (
    HAZARDS_PARQUET,
    SAFE_ZONES_GEOJSON,
    ECOSYSTEM_JSON,
    RAKSHAK_RUN_JSON,
)
from apps.api.leher.jobs.rakshak_job import execute_rakshak_job, _rakshak_lock

router = APIRouter(tags=["ML"])

@router.get("/events")
def get_events():
    if not HAZARDS_PARQUET.exists():
        raise HTTPException(status_code=404, detail="Hazards data not found")
    try:
        df = pl.read_parquet(HAZARDS_PARQUET)
        return JSONResponse(content=df.to_dicts())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/{event_id}")
def get_event(event_id: str):
    if not HAZARDS_PARQUET.exists():
        raise HTTPException(status_code=404, detail="Hazards data not found")
    try:
        df = pl.read_parquet(HAZARDS_PARQUET)
        event_df = df.filter(pl.col("event_id") == event_id)
        if len(event_df) == 0:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
        return JSONResponse(content=event_df.to_dicts()[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/zones")
def get_zones():
    if not SAFE_ZONES_GEOJSON.exists():
        raise HTTPException(status_code=404, detail="Safe zones data not found")
    return FileResponse(SAFE_ZONES_GEOJSON, media_type="application/geo+json")

@router.get("/ecosystem")
def get_ecosystem():
    if not ECOSYSTEM_JSON.exists():
        raise HTTPException(status_code=404, detail="Ecosystem data not found")
    try:
        with open(ECOSYSTEM_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/run-status")
def get_run_status():
    if not RAKSHAK_RUN_JSON.exists():
        raise HTTPException(status_code=404, detail="Run status not found")
    try:
        with open(RAKSHAK_RUN_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run")
def trigger_run(request: Request):
    if _rakshak_lock.locked():
        return JSONResponse(
            content={"status": "skipped", "reason": "rakshak_run_already_in_progress"}
        )
    
    scheduler = request.app.state.scheduler
    scheduler.add_job(
        execute_rakshak_job,
        id="rakshak_manual_run",
        replace_existing=True
    )
    return JSONResponse(content={"status": "accepted"})
