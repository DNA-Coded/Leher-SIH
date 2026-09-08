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
from apps.api.leher.jobs.rakshak_job import execute_rakshak_job, _rakshak_lock, get_execution_status

router = APIRouter(tags=["ML"])

@router.get("/events")
def get_events():
    if not HAZARDS_PARQUET.exists():
        raise HTTPException(status_code=503, detail="ML artifacts unavailable - Rakshak scan has not completed")
    try:
        df = pl.read_parquet(HAZARDS_PARQUET)
        # Convert NaN to null for strict JSON compliance in FastAPI
        df = df.with_columns([pl.col(c).fill_nan(None) for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)])
        return JSONResponse(content=df.to_dicts())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/events/{event_id}")
def get_event(event_id: str):
    if not HAZARDS_PARQUET.exists():
        raise HTTPException(status_code=503, detail="ML artifacts unavailable - Rakshak scan has not completed")
    try:
        df = pl.read_parquet(HAZARDS_PARQUET)
        event_df = df.filter(pl.col("event_id") == event_id)
        if len(event_df) == 0:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
            
        event_df = event_df.with_columns([pl.col(c).fill_nan(None) for c in event_df.columns if event_df[c].dtype in (pl.Float32, pl.Float64)])
        return JSONResponse(content=event_df.to_dicts()[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/zones")
def get_zones():
    if not SAFE_ZONES_GEOJSON.exists():
        raise HTTPException(status_code=503, detail="ML artifacts unavailable - Rakshak scan has not completed")
    return FileResponse(SAFE_ZONES_GEOJSON, media_type="application/geo+json")

@router.get("/ecosystem")
def get_ecosystem():
    if not ECOSYSTEM_JSON.exists():
        raise HTTPException(status_code=503, detail="ML artifacts unavailable - Rakshak scan has not completed")
    try:
        with open(ECOSYSTEM_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/run-status")
def get_run_status():
    status = {
        "execution": get_execution_status(),
        "last_completed_run": None
    }
    
    if RAKSHAK_RUN_JSON.exists():
        try:
            with open(RAKSHAK_RUN_JSON, "r", encoding="utf-8") as f:
                status["last_completed_run"] = json.load(f)
        except Exception:
            pass
            
    return JSONResponse(content=status)

from fastapi.responses import Response
from apps.api.leher.core.arrow_encoder import encode_polars_to_arrow_ipc

@router.get("/hazards")
def get_hazards():
    if not HAZARDS_PARQUET.exists():
        raise HTTPException(status_code=503, detail="ML artifacts unavailable - Rakshak scan has not completed")
    try:
        df = pl.read_parquet(HAZARDS_PARQUET)
        ipc_bytes = encode_polars_to_arrow_ipc(df)
        return Response(
            content=ipc_bytes,
            media_type="application/vnd.apache.arrow.stream"
        )
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
