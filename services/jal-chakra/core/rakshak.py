import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
rakshak.py — Full Rakshak Intelligence Inference Pipeline
Leher / Rakshak Intelligence

This is what the APScheduler triggers every hour in production.
It runs ALL Rakshak models on the latest available data and writes:
  - fabric/ml/hazards.parquet        (cyclone + surge events)
  - fabric/ml/safe_zones.geojson     (Green/Yellow/Red fishing zones)
  - fabric/ml/safe_zones.parquet     (same, for DuckDB queries)
  - fabric/ml/rakshak_run.json       (run metadata for the backend)

The backend developer calls:
  from services.jal_chakra.core.rakshak import run_rakshak
  result = run_rakshak()
"""
import json
import numpy as np
import polars as pl
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone

# Support both: `python rakshak.py` (standalone) and `from .rakshak import run_rakshak` (module)
if __package__:
    from .safe_zones import (
        load_surface_snapshot,
        coarsen_to_grid,
        compute_safe_zones,
        zones_to_geojson,
    )
else:
    from pathlib import Path as _P
    import importlib.util as _ilu
    _sz_path = _P(__file__).parent / "safe_zones.py"
    _spec = _ilu.spec_from_file_location("safe_zones", _sz_path)
    _sz = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_sz)
    load_surface_snapshot = _sz.load_surface_snapshot
    coarsen_to_grid       = _sz.coarsen_to_grid
    compute_safe_zones    = _sz.compute_safe_zones
    zones_to_geojson      = _sz.zones_to_geojson

if __package__:
    from .tides import detect_extreme_tides
    from .voice_advisory import generate_advisories
else:
    _t_path = _P(__file__).parent / "tides.py"
    _t_spec = _ilu.spec_from_file_location("tides", _t_path)
    _t = _ilu.module_from_spec(_t_spec)
    _t_spec.loader.exec_module(_t)
    detect_extreme_tides = _t.detect_extreme_tides
    
    _v_path = _P(__file__).parent / "voice_advisory.py"
    _v_spec = _ilu.spec_from_file_location("voice_advisory", _v_path)
    _v = _ilu.module_from_spec(_v_spec)
    _v_spec.loader.exec_module(_v)
    generate_advisories = _v.generate_advisories

PARQUET_DIR    = Path("data/ml_training_dataset.parquet")
CYCLONE_MODEL  = Path("fabric/ml/anomaly_model.json")
SURGE_MODEL    = Path("fabric/ml/surge_model.json")
HAZARDS_OUT    = Path("fabric/ml/hazards.parquet")
ZONES_GEOJSON  = Path("fabric/ml/safe_zones.geojson")
ZONES_PARQUET  = Path("fabric/ml/safe_zones.parquet")
RUN_META_OUT   = Path("fabric/ml/rakshak_run.json")

# Cyclone detection threshold (from LeherGuide.md Section 21)
CYCLONE_THRESHOLD = 0.65

# Feature columns expected by the cyclone model
CYCLONE_FEATURES = [
    "sst", "temp_100m", "thermal_contrast",
    "current_speed", "surface_uo", "surface_vo",
    "vorticity", "zos", "mlotst",
]

# Feature columns expected by the surge model
SURGE_FEATURES = [
    "sst", "current_speed", "surface_uo", "surface_vo",
    "coastal_proximity", "zos",
]


def _load_model(path: Path, model_class):
    """Load a saved XGBoost model, raising clearly if not found."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "  Train it first:\n"
            f"  {'train_cyclone_model.py' if 'anomaly' in path.stem else 'train_surge_model.py'}"
        )
    model = model_class()
    model.load_model(str(path))
    print(f"[RAKSHAK] Loaded model: {path.name}")
    return model


def build_inference_features(surface_df: pl.DataFrame) -> pl.DataFrame:
    """
    Build the feature vector for every grid cell from the surface snapshot.
    Mirrors what feature_engineering.py does, but for the latest data only.
    """
    # We don't have 100m data in the surface snapshot, so thermal_contrast = NaN
    # XGBoost handles NaN natively.
    df = surface_df.with_columns([
        pl.col("sst").alias("sst"),
        pl.lit(float("nan")).alias("temp_100m"),
        pl.lit(float("nan")).alias("thermal_contrast"),
        (pl.col("uo") ** 2 + pl.col("vo") ** 2).sqrt().alias("current_speed"),
        pl.col("uo").alias("surface_uo"),
        pl.col("vo").alias("surface_vo"),
        pl.lit(0.0).alias("vorticity"),      # simplified
        pl.lit(float("nan")).alias("zos"),   # not in training parquet
        pl.lit(float("nan")).alias("mlotst"),# not in training parquet
    ])

    # Coastal proximity for surge model
    df = df.with_columns([
        pl.when(
            (pl.col("lat_grid").is_between(8.0, 22.0)) &
            (pl.col("lon_grid").is_between(80.0, 100.0))
        )
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.3))
        .alias("coastal_proximity")
    ])

    return df


def run_cyclone_scan(
    model: xgb.XGBClassifier,
    features_df: pl.DataFrame,
) -> list[dict]:
    """
    Run cyclone detection on every grid cell.
    Returns a list of hazard events (cells where P(cyclone) > threshold).
    """
    X = features_df.select(CYCLONE_FEATURES).to_numpy(allow_copy=True).astype(np.float32)

    # Batch predict — much faster than row-by-row
    probs = model.predict_proba(X)[:, 1]  # P(cyclone)
    timestamp = datetime.now(timezone.utc).isoformat()

    hazards = []
    event_id = 0
    for idx, prob in enumerate(probs):
        if prob >= CYCLONE_THRESHOLD:
            event_id += 1
            row = features_df.row(idx, named=True)
            hazards.append({
                "event_id": f"CYC-{event_id:04d}",
                "type": "cyclone",
                "latitude": float(row["lat_grid"]),
                "longitude": float(row["lon_grid"]),
                "probability": round(float(prob), 4),
                "sst": round(float(row["sst"]), 2) if not np.isnan(row["sst"]) else None,
                "current_speed": round(float(row["current_speed"]), 3),
                "detected_at": timestamp,
            })

    print(f"[RAKSHAK] Cyclone scan: {len(hazards)} event(s) detected (threshold={CYCLONE_THRESHOLD})")
    return hazards


def run_surge_scan(
    model: xgb.XGBRegressor,
    features_df: pl.DataFrame,
) -> list[dict]:
    """
    Run storm surge prediction on coastal cells.
    Returns cells with predicted surge height > 0.5m.
    """
    coastal = features_df.filter(pl.col("coastal_proximity") >= 0.3)
    if len(coastal) == 0:
        print("[RAKSHAK] Surge scan: no coastal cells found")
        return []

    X = coastal.select(SURGE_FEATURES).to_numpy(allow_copy=True).astype(np.float32)
    surge_heights = model.predict(X)
    timestamp = datetime.now(timezone.utc).isoformat()

    surge_events = []
    event_id = 0
    for idx, height in enumerate(surge_heights):
        if height > 0.5:  # 0.5m threshold from LeherGuide.md Section 21.3
            event_id += 1
            row = coastal.row(idx, named=True)
            surge_events.append({
                "event_id": f"SURGE-{event_id:04d}",
                "type": "storm_surge",
                "latitude": float(row["lat_grid"]),
                "longitude": float(row["lon_grid"]),
                "surge_height_m": round(float(height), 3),
                "current_speed": round(float(row["current_speed"]), 3),
                "detected_at": timestamp,
            })

    print(f"[RAKSHAK] Surge scan: {len(surge_events)} coastal cell(s) with surge > 0.5m")
    return surge_events


def run_rakshak(
    parquet_dir: Path = PARQUET_DIR,
    cyclone_model_path: Path = CYCLONE_MODEL,
    surge_model_path: Path = SURGE_MODEL,
) -> dict:
    """
    Full Rakshak inference pipeline.
    Safe to call even if surge model is not yet trained (degrades gracefully).

    Returns a summary dict for the backend to read via rakshak_run.json.
    """
    print()
    print("=" * 55)
    print("RAKSHAK INTELLIGENCE SCAN")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 55)

    # Step 1: Load latest surface data
    surface_df = load_surface_snapshot()
    grid_df = coarsen_to_grid(surface_df)

    # Step 2: Build feature vectors
    features_df = build_inference_features(grid_df)

    all_hazards = []

    # Step 3: Cyclone scan
    if cyclone_model_path.exists():
        cyclone_model = _load_model(cyclone_model_path, xgb.XGBClassifier)
        cyclone_events = run_cyclone_scan(cyclone_model, features_df)
        all_hazards.extend(cyclone_events)
    else:
        print(f"[RAKSHAK] Cyclone model not found — skipping. Train it first.")
        cyclone_events = []

    # Step 4: Surge scan
    if surge_model_path.exists():
        surge_model = _load_model(surge_model_path, xgb.XGBRegressor)
        surge_events = run_surge_scan(surge_model, features_df)
        all_hazards.extend(surge_events)
    else:
        print(f"[RAKSHAK] Surge model not found — skipping. Train it first.")
        surge_events = []

    # Step 4.5: Extreme Tide scan (Engine 3)
    tide_events = detect_extreme_tides(grid_df)
    all_hazards.extend(tide_events)

    # Step 5: Write hazards
    HAZARDS_OUT.parent.mkdir(parents=True, exist_ok=True)
    if all_hazards:
        hazards_df = pl.DataFrame(all_hazards)
        hazards_df.write_parquet(HAZARDS_OUT)
        print(f"[RAKSHAK] {len(all_hazards)} hazard(s) written to {HAZARDS_OUT}")
    else:
        # Write empty file so backend doesn't get a FileNotFoundError
        pl.DataFrame({"event_id": [], "type": [], "latitude": [], "longitude": []}).write_parquet(HAZARDS_OUT)
        print("[RAKSHAK] No hazards — Indian Ocean is calm")

    # Step 6: Compute safe zones (uses hazards for cyclone proximity)
    print("[RAKSHAK] Computing fishing zone classification...")
    zones = compute_safe_zones(grid_df, all_hazards)
    geojson = zones_to_geojson(zones)
    ZONES_GEOJSON.write_text(json.dumps(geojson))
    pl.DataFrame(zones).write_parquet(ZONES_PARQUET)
    print(f"[RAKSHAK] {len(zones)} zones -> {ZONES_GEOJSON}")
    
    # Step 6.5: Voice Advisories (Engine 5)
    advisories = generate_advisories(all_hazards)

    # Step 7: Write run metadata
    result = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "n_cyclone_events": len(cyclone_events),
        "n_surge_events": len(surge_events),
        "n_tide_events": len(tide_events),
        "n_zones": len(zones),
        "n_safe": geojson["metadata"]["n_safe"],
        "n_caution": geojson["metadata"]["n_caution"],
        "n_danger": geojson["metadata"]["n_danger"],
        "outputs": {
            "hazards": str(HAZARDS_OUT),
            "safe_zones_geojson": str(ZONES_GEOJSON),
            "safe_zones_parquet": str(ZONES_PARQUET),
        },
    }
    RUN_META_OUT.write_text(json.dumps(result, indent=2))

    print()
    print("=" * 55)
    print("RAKSHAK SCAN COMPLETE")
    print(f"  Cyclone events : {len(cyclone_events)}")
    print(f"  Surge events   : {len(surge_events)}")
    print(f"  Tide events    : {len(tide_events)}")
    print(f"  Safe zones     : {geojson['metadata']['n_safe']}")
    print(f"  Caution zones  : {geojson['metadata']['n_caution']}")
    print(f"  Danger zones   : {geojson['metadata']['n_danger']}")
    print(f"  Voice Warnings : {len(advisories)}")
    print("=" * 55)

    return result


if __name__ == "__main__":
    run_rakshak()
