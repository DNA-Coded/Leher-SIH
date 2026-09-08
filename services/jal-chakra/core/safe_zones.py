"""
safe_zones.py — Fisherman Safe / Caution / Danger Zone Engine
Leher / Rakshak Intelligence

Works directly on the existing parquet data (no Zarr needed).
Computes a risk score for each 0.25° x 0.25° grid cell.

Risk Categories:
  Green  (Safe)    : risk_score < 0.3
  Yellow (Caution) : 0.3 <= risk_score < 0.7
  Red    (Danger)  : risk_score >= 0.7

Risk factors (from LeherGuide.md Section 22):
  - current_speed > 1.5 m/s -> +0.5
  - current_speed > 0.5 m/s -> +0.2
  - SST > 29.0°C            -> +0.2 (cyclone-favorable warmth)
  - thermal_contrast > 8°C  -> +0.15 (atmospheric instability)
  - active cyclone within 300km -> +0.5 (from hazards.parquet)
  - storm surge prob > 0.5  -> +0.3
"""
import json
import numpy as np
import polars as pl
from pathlib import Path


PARQUET_DIR  = Path("data/ml_training_dataset.parquet")
HAZARDS_PATH = Path("fabric/ml/hazards.parquet")
OUTPUT_GEOJSON = Path("fabric/ml/safe_zones.geojson")
OUTPUT_PARQUET = Path("fabric/ml/safe_zones.parquet")

# Indian Ocean bounding box
LAT_MIN, LAT_MAX = -20.0, 25.0
LON_MIN, LON_MAX = 50.0, 100.0

# Grid cell size for zone output
GRID_DEG = 0.25


def _haversine_deg(lat1, lon1, lat2, lon2) -> float:
    """Approximate distance in degrees (fast, used for cyclone proximity)."""
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    return np.sqrt(dlat**2 + dlon**2)


def load_surface_snapshot() -> pl.DataFrame:
    """
    Load the surface layer (depth <= 5m) from the parquet dataset.
    Uses the most recent timestamp available.
    """
    print("[ZONES] Loading surface snapshot from parquet...")
    lf = pl.scan_parquet(str(PARQUET_DIR / "*.parquet"))

    # Get most recent time
    latest_time = lf.select(pl.col("time").max()).collect().item()
    print(f"  Most recent timestamp: {latest_time}")

    # Surface layer at the latest time
    df = (
        lf.filter(
            (pl.col("depth") <= 5.0) &
            (pl.col("time") == latest_time)
        )
        .select(["latitude", "longitude", "thetao", "uo", "vo", "surface_temp"])
        .collect()
    )
    print(f"  Surface cells loaded: {len(df):,}")
    return df


def coarsen_to_grid(df: pl.DataFrame, grid_deg: float = GRID_DEG) -> pl.DataFrame:
    """
    Coarsen the fine-resolution data to a coarser grid for zone output.
    Rounds lat/lon to nearest grid_deg and takes the mean within each cell.
    """
    df = df.with_columns([
        (pl.col("latitude") / grid_deg).round(0).cast(pl.Int32) * grid_deg,
        (pl.col("longitude") / grid_deg).round(0).cast(pl.Int32) * grid_deg,
    ])
    # Rename after rounding
    df = df.rename({"latitude": "lat_grid", "longitude": "lon_grid"})

    df = (
        df.group_by(["lat_grid", "lon_grid"])
        .agg([
            pl.col("thetao").mean().alias("sst"),
            pl.col("uo").mean().alias("uo"),
            pl.col("vo").mean().alias("vo"),
        ])
        .with_columns([
            (pl.col("uo") ** 2 + pl.col("vo") ** 2).sqrt().alias("current_speed"),
        ])
    )
    print(f"  Coarsened to {len(df):,} grid cells at {grid_deg}° resolution")
    return df


def load_active_hazards() -> list[dict]:
    """Load hazards from Rakshak inference output, if available."""
    if not HAZARDS_PATH.exists():
        print("[ZONES] No hazards.parquet found — no cyclone proximity adjustment")
        return []
    df = pl.read_parquet(HAZARDS_PATH)
    hazards = df.to_dicts()
    print(f"[ZONES] Loaded {len(hazards)} active hazard(s)")
    return hazards


def compute_risk_score(
    sst: float,
    current_speed: float,
    thermal_contrast: float,
    cyclone_min_dist_deg: float,
) -> tuple[float, list[str]]:
    """
    Compute a risk score [0, 1] for a single grid cell.
    Returns (risk_score, list_of_contributing_factors).
    """
    risk = 0.0
    factors = []

    # Current speed
    if current_speed > 1.5:
        risk += 0.5
        factors.append("high_current_speed")
    elif current_speed > 0.5:
        risk += 0.2
        factors.append("moderate_current_speed")

    # SST threshold for cyclone development
    if sst > 29.0:
        risk += 0.2
        factors.append("elevated_sst")

    # Cyclone proximity: 300km ≈ 2.7° latitude
    if cyclone_min_dist_deg < 2.7:
        risk += 0.5
        factors.append("active_cyclone_nearby")
    elif cyclone_min_dist_deg < 5.4:  # 600km
        risk += 0.2
        factors.append("cyclone_in_region")

    return min(risk, 1.0), factors


def compute_safe_zones(
    surface_df: pl.DataFrame,
    hazards: list[dict],
) -> list[dict]:
    """
    Compute zone classification for every grid cell in the dataset.
    Returns a list of dicts ready for GeoJSON serialization.
    """
    print("[ZONES] Computing risk scores...")

    # Pre-compute cyclone locations
    cyclone_lats = np.array([h["latitude"] for h in hazards]) if hazards else np.array([])
    cyclone_lons = np.array([h["longitude"] for h in hazards]) if hazards else np.array([])

    rows = surface_df.to_dicts()
    zones = []

    for row in rows:
        lat = row["lat_grid"]
        lon = row["lon_grid"]
        sst = row.get("sst", float("nan"))
        speed = row.get("current_speed", float("nan"))

        if np.isnan(sst) or np.isnan(speed):
            continue  # Land or missing data — skip

        thermal_contrast = 0.0  # Not available in this surface-only snapshot

        # Cyclone proximity
        if len(cyclone_lats) > 0:
            dists = np.sqrt((lat - cyclone_lats)**2 + (lon - cyclone_lons)**2)
            min_dist = float(dists.min())
        else:
            min_dist = 999.0  # No active cyclones

        risk, factors = compute_risk_score(sst, speed, thermal_contrast, min_dist)

        category = (
            "safe" if risk < 0.3
            else "caution" if risk < 0.7
            else "danger"
        )

        zones.append({
            "lat": lat,
            "lon": lon,
            "risk_score": round(risk, 3),
            "risk_category": category,
            "sst": round(float(sst), 2) if not np.isnan(sst) else None,
            "current_speed": round(float(speed), 3),
            "factors": factors,
        })

    # Summary
    n_safe    = sum(1 for z in zones if z["risk_category"] == "safe")
    n_caution = sum(1 for z in zones if z["risk_category"] == "caution")
    n_danger  = sum(1 for z in zones if z["risk_category"] == "danger")
    print(f"  Safe: {n_safe:,}  Caution: {n_caution:,}  Danger: {n_danger:,}")

    return zones


def zones_to_geojson(zones: list[dict]) -> dict:
    """Convert zone list to a GeoJSON FeatureCollection."""
    color_map = {"safe": "#00C853", "caution": "#FFD600", "danger": "#D50000"}

    features = []
    for z in zones:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [z["lon"], z["lat"]],
            },
            "properties": {
                "risk_score": z["risk_score"],
                "risk_category": z["risk_category"],
                "sst": z["sst"],
                "current_speed": z["current_speed"],
                "factors": z["factors"],
                "color": color_map.get(z["risk_category"], "#888888"),
            },
        })
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_cells": len(zones),
            "n_safe": sum(1 for z in zones if z["risk_category"] == "safe"),
            "n_caution": sum(1 for z in zones if z["risk_category"] == "caution"),
            "n_danger": sum(1 for z in zones if z["risk_category"] == "danger"),
            "grid_resolution_deg": GRID_DEG,
        },
    }


def run_safe_zone_computation() -> Path:
    """
    Full pipeline:
      1. Load surface snapshot from parquet
      2. Coarsen to 0.25° grid
      3. Load active hazards
      4. Compute risk scores
      5. Save GeoJSON + Parquet
    """
    surface = load_surface_snapshot()
    grid = coarsen_to_grid(surface)
    hazards = load_active_hazards()
    zones = compute_safe_zones(grid, hazards)

    # Save GeoJSON (for Cesium frontend)
    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    geojson = zones_to_geojson(zones)
    OUTPUT_GEOJSON.write_text(json.dumps(geojson))
    print(f"[ZONES] GeoJSON saved: {OUTPUT_GEOJSON} ({OUTPUT_GEOJSON.stat().st_size / 1024:.0f} KB)")

    # Save Parquet (for DuckDB / FastAPI)
    zones_df = pl.DataFrame(zones)
    zones_df.write_parquet(OUTPUT_PARQUET)
    print(f"[ZONES] Parquet saved: {OUTPUT_PARQUET}")

    return OUTPUT_GEOJSON


if __name__ == "__main__":
    run_safe_zone_computation()
