import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response
from pathlib import Path
import sys
import json
import numpy as np

from apps.api.leher.core.arrow_encoder import encode_dataset_to_arrow_ipc
from apps.api.leher.paths import REPO_ROOT

_jal_chakra_dir = str(REPO_ROOT / "services" / "jal-chakra")
if _jal_chakra_dir not in sys.path:
    sys.path.insert(0, _jal_chakra_dir)

from core.samudra_reader import (
    get_active_glorys_dataset,
    read_samudra_slice,
    NoActiveDatasetError,
    MultipleActiveDatasetsError,
    MissingZarrVariableError,
    InvalidDepthError,
    InvalidTimeError
)

router = APIRouter(tags=["Model Slices"])

# A simple in-memory TTL cache for slice requests
_slice_cache = {}

from apps.api.leher.config import CACHE_TTL_SECONDS as CACHE_TTL

@router.get("/slices", responses={200: {"content": {"application/vnd.apache.arrow.stream": {}}}})
def get_model_slice(
    dataset_id: Optional[str] = Query(None, description="Dataset revision. If omitted, uses active dataset."),
    variable: str = Query(..., description="Canonical variable name (e.g., temperature, salinity)"),
    depth_m: float = Query(0.0, description="Positive-down depth in meters"),
    time_str: str = Query("latest", alias="time", description="Timestamp or 'latest'"),
    min_lat: float = Query(..., description="Minimum latitude bounding box"),
    max_lat: float = Query(..., description="Maximum latitude bounding box"),
    min_lon: float = Query(..., description="Minimum longitude bounding box"),
    max_lon: float = Query(..., description="Maximum longitude bounding box"),
    resolution: float = Query(0.25, description="Requested spatial grid resolution (informational only for current version; data is returned at native model resolution)")
):
    # Validation
    if depth_m < 0:
        raise HTTPException(status_code=400, detail="Depth must be >= 0 (positive-down)")
    if min_lat >= max_lat or min_lon >= max_lon:
        raise HTTPException(status_code=400, detail="Invalid bounding box coordinates")
    if resolution <= 0:
        raise HTTPException(status_code=400, detail="Resolution must be positive")
        
    cache_key = (dataset_id, variable, depth_m, time_str, min_lat, max_lat, min_lon, max_lon, resolution)
    now = time.time()
    
    # Cache lookup
    if cache_key in _slice_cache:
        cached_time, cached_payload, cached_headers = _slice_cache[cache_key]
        if now - cached_time < CACHE_TTL:
            return Response(content=cached_payload, media_type="application/vnd.apache.arrow.stream", headers=cached_headers)
            
    fabric_dir = REPO_ROOT / "fabric"
    
    try:
        if dataset_id:
            rev_dir = fabric_dir / "datasets" / "glorys" / dataset_id
            if not rev_dir.exists():
                raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
                
            manifest_path = rev_dir / "manifest.json"
            if not manifest_path.exists():
                raise HTTPException(status_code=404, detail=f"Manifest not found for {dataset_id}")
                
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            dataset_path = rev_dir
            active_dataset_id = dataset_id
        else:
            dataset_path, manifest = get_active_glorys_dataset(fabric_dir)
            active_dataset_id = manifest.get("dataset_id", "unknown")
            
        # Verify variable exists in manifest
        available_vars = [v.get("canonical_name") for v in manifest.get("variables", [])]
        if variable not in available_vars:
            raise HTTPException(status_code=400, detail=f"Unsupported variable '{variable}'. Available: {available_vars}")
            
        # Read the slice
        ds = read_samudra_slice(
            dataset_path=dataset_path,
            manifest=manifest,
            variables=[variable],
            depth_m=depth_m,
            time_str=time_str,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon
        )
        
        # Determine actual selected depth
        if "depth" in ds.coords:
            actual_depth = float(ds.depth.values)
        else:
            actual_depth = depth_m
            
        if "time" in ds.coords:
            actual_time = str(ds.time.values)
        else:
            actual_time = time_str
            
        # Check actual bounds returned
        lats = ds.latitude.values if "latitude" in ds.coords else []
        lons = ds.longitude.values if "longitude" in ds.coords else []
        rows = len(lats)
        cols = len(lons)
        
        # Calculate min/max across the slice safely
        var_data = ds[variable].values
        # Ignoring NaNs for min/max
        data_min = float(np.nanmin(var_data)) if var_data.size > 0 and not np.all(np.isnan(var_data)) else float('nan')
        data_max = float(np.nanmax(var_data)) if var_data.size > 0 and not np.all(np.isnan(var_data)) else float('nan')
        
        payload = encode_dataset_to_arrow_ipc(ds, variable)
        
        # Create metadata headers
        headers = {
            "X-Dataset-Id": active_dataset_id,
            "X-Selected-Variable": variable,
            "X-Selected-Time": actual_time,
            "X-Selected-Depth": str(actual_depth),
            "X-Slice-Rows": str(rows),
            "X-Slice-Cols": str(cols),
            "X-Data-Min": str(data_min),
            "X-Data-Max": str(data_max)
        }
        
        _slice_cache[cache_key] = (now, payload, headers)
        
        return Response(content=payload, media_type="application/vnd.apache.arrow.stream", headers=headers)
        
    except (NoActiveDatasetError, MultipleActiveDatasetsError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except MissingZarrVariableError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (InvalidDepthError, InvalidTimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
