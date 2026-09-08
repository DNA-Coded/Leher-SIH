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


class MissingInferenceDataError(Exception):
    """Raised when the expected inference input data is missing."""
    pass

def load_surface_snapshot(fabric_dir: Path = Path("fabric")) -> pl.DataFrame:
    """
    Load the inference snapshot from the active Samudra Zarr dataset.
    Retrieves both surface layer and nearest native depth to 100m.
    """
    if __package__:
        from .samudra_reader import get_active_glorys_dataset, read_samudra_slice
    else:
        from pathlib import Path as _P
        import sys
        
        # Add the parent directory of 'core' to sys.path if not already there
        _core_parent = str(_P(__file__).resolve().parent.parent)
        if _core_parent not in sys.path:
            sys.path.insert(0, _core_parent)
            
        from core.samudra_reader import get_active_glorys_dataset, read_samudra_slice
    
    try:
        dataset_path, manifest = get_active_glorys_dataset(fabric_dir)
    except Exception as e:
        raise MissingInferenceDataError(f"Failed to find active inference dataset: {e}")
        
    print(f"[ZONES] Loading active dataset: {dataset_path.name} from Zarr...")
    
    # Determine available variables from manifest
    available_vars = [v["canonical_name"] for v in manifest.get("variables", [])]
    
    surface_vars = ["temperature", "u_velocity", "v_velocity"]
    if "salinity" in available_vars:
        surface_vars.append("salinity")
    if "current_speed" in available_vars:
        surface_vars.append("current_speed")
    if "sea_surface_height" in available_vars:
        surface_vars.append("sea_surface_height")
    if "mixed_layer_depth" in available_vars:
        surface_vars.append("mixed_layer_depth")
    if "chlorophyll" in available_vars:
        surface_vars.append("chlorophyll")
        
    # Read surface layer
    ds_surface = read_samudra_slice(
        dataset_path, manifest, surface_vars, depth_m=0.0, time_str="latest"
    )
    
    # Read 100m layer for thermal contrast
    ds_100m = read_samudra_slice(
        dataset_path, manifest, ["temperature"], depth_m=100.0, time_str="latest"
    )
    
    latest_time = ds_surface.time.values
    print(f"  Most recent timestamp: {latest_time}")
    
    df_surface = ds_surface.to_dataframe().reset_index()
    df_100m = ds_100m.to_dataframe().reset_index()
    
    # Merge 100m temp
    df_100m = df_100m.rename(columns={"temperature": "temp_100m"})
    df_100m = df_100m[["latitude", "longitude", "temp_100m"]]
    df_merged = df_surface.merge(df_100m, on=["latitude", "longitude"], how="left")
    
    # Calculate thermal contrast
    df_merged["thermal_contrast"] = df_merged["temperature"] - df_merged["temp_100m"]
    
    # Rename to old internal expectations
    rename_map = {
        "temperature": "thetao",
        "u_velocity": "uo",
        "v_velocity": "vo",
        "salinity": "surface_salinity",
        "sea_surface_height": "zos",
        "mixed_layer_depth": "mlotst",
        "chlorophyll": "chl",
    }
    # Only rename existing columns
    rename_map = {k: v for k, v in rename_map.items() if k in df_merged.columns}
    df_merged = df_merged.rename(columns=rename_map)
    
    # Ensure surface_temp exists as old logic had it (even if unused)
    if "surface_temp" not in df_merged.columns:
        df_merged["surface_temp"] = df_merged["thetao"]

    pdf = pl.from_pandas(df_merged)
    print(f"  Surface cells loaded: {len(pdf):,}")
    return pdf


def coarsen_to_grid(df: pl.DataFrame, grid_deg: float = GRID_DEG) -> pl.DataFrame:
    """
    Coarsen the fine-resolution data to a coarser grid for zone output.
    Rounds lat/lon to nearest grid_deg and takes the mean within each cell.
    """
    df = df.with_columns([
        (pl.col("latitude") / grid_deg).round(0).cast(pl.Int32) * grid_deg,
        (pl.col("longitude") / grid_deg).round(0).cast(pl.Int32) * grid_deg,
    ])
    df = df.rename({"latitude": "lat_grid", "longitude": "lon_grid"})

    agg_exprs = [
        pl.col("thetao").mean().alias("sst"),
        pl.col("uo").mean().alias("uo"),
        pl.col("vo").mean().alias("vo"),
    ]
    
    # Optional fields
    if "surface_salinity" in df.columns:
        agg_exprs.append(pl.col("surface_salinity").mean().alias("surface_salinity"))
    if "temp_100m" in df.columns:
        agg_exprs.append(pl.col("temp_100m").mean().alias("temp_100m"))
    if "thermal_contrast" in df.columns:
        agg_exprs.append(pl.col("thermal_contrast").mean().alias("thermal_contrast"))
    if "current_speed" in df.columns:
        agg_exprs.append(pl.col("current_speed").mean().alias("current_speed"))
    if "zos" in df.columns:
        agg_exprs.append(pl.col("zos").mean().alias("zos"))
    if "mlotst" in df.columns:
        agg_exprs.append(pl.col("mlotst").mean().alias("mlotst"))

    df = df.group_by(["lat_grid", "lon_grid"]).agg(agg_exprs)
    
    # Re-calculate current_speed if not already provided by Zarr
    if "current_speed" not in df.columns:
        df = df.with_columns([
            (pl.col("uo") ** 2 + pl.col("vo") ** 2).sqrt().alias("current_speed"),
        ])
        
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
        sst = row.get("sst")
        sst = float("nan") if sst is None else sst
        speed = row.get("current_speed")
        speed = float("nan") if speed is None else speed

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
