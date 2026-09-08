"""
tides.py — Extreme Tide & Anomaly Residual Detection Engine
Leher / Rakshak Intelligence (Engine 3)

Physics/Harmonic Rule-Based Engine.
Detects abnormal tidal flooding caused by astronomical spring tides 
compounded by thermal expansion and wind setup.

Input: features_unlabeled.parquet
Output: extreme_tides list (merged into hazards.parquet by Rakshak)
"""
import numpy as np
import polars as pl
from datetime import datetime, timezone

def detect_extreme_tides(surface_df: pl.DataFrame) -> list[dict]:
    """
    Detects extreme tide residuals > 0.3m.
    
    Since real `zos` (sea surface height) from Copernicus is pending download,
    this uses a physics-based proxy for now (thermal expansion + current piling).
    """
    print("[TIDES] Running Extreme Tide & Anomaly Detection...")
    
    # Filter to coastal cells only (approximate via bounding box)
    coastal = surface_df.filter(
        (pl.col("lat_grid").is_between(8.0, 25.0)) & 
        (pl.col("lon_grid").is_between(65.0, 95.0))
    )
    
    if len(coastal) == 0:
        return []

    # Calculate tidal residual proxy
    # proxy_residual = (current_speed * 0.1) + (sst anomaly * 0.05)
    # When real zos is available, this will be: zos - astronomical_tide
    if "zos" not in coastal.columns:
        coastal = coastal.with_columns(pl.lit(float("nan")).alias("zos"))
        
    df = coastal.with_columns([
        pl.when(pl.col("zos").is_not_nan() & pl.col("zos").is_not_null())
        .then(pl.col("zos"))  # Use real zos if available
        .otherwise(
            (pl.col("current_speed") * 0.1) + ((pl.col("sst") - 28.0).clip(0.0, 5.0) * 0.05)
        ).alias("tidal_residual_m")
    ])

    # Filter residuals > 0.3m (LeherGuide threshold)
    extreme_tides = df.filter(pl.col("tidal_residual_m") > 0.3)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    events = []
    event_id = 0
    
    for row in extreme_tides.iter_rows(named=True):
        event_id += 1
        events.append({
            "event_id": f"TIDE-{event_id:04d}",
            "type": "extreme_tide",
            "latitude": float(row["lat_grid"]),
            "longitude": float(row["lon_grid"]),
            "residual_height_m": round(float(row["tidal_residual_m"]), 3),
            "sst": round(float(row["sst"]), 2) if row["sst"] is not None and not np.isnan(row["sst"]) else None,
            "detected_at": timestamp,
        })

    print(f"[TIDES] Detected {len(events)} extreme tide anomalies (> 0.3m residual)")
    return events
