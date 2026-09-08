"""
feature_engineering.py — ML Training Feature Builder
Leher / Rakshak Intelligence

Builds the training feature set from the existing parquet dataset at:
  data/ml_training_dataset.parquet/

Available columns:
  time, depth, latitude, longitude, thetao, so, uo, vo, surface_temp, surface_salinity

What we derive:
  - surface features (depth <= 5m)
  - thermal_contrast (surface - 100m temp)
  - current_speed = sqrt(uo^2 + vo^2)
  - vorticity approximation (spatial gradient of uo, vo at surface)
  - salinity anomaly from mean

What is NOT in the data (will be filled with NaN placeholders):
  - zos (sea surface height) — needs separate Copernicus download
  - mlotst (mixed layer depth) — needs separate Copernicus download

After IBTrACS labels are joined, this dataset trains the XGBoost model.
"""
import numpy as np
import polars as pl
from pathlib import Path


DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "ml_training_dataset.parquet"
OUTPUT_DIR = Path("fabric/ml")


def load_parquet_dataset() -> pl.LazyFrame:
    """Load all 44 parquet parts as a lazy frame (memory-efficient)."""
    print("[FEATURE ENG] Loading parquet dataset (lazy)...")
    lf = pl.scan_parquet(str(PARQUET_DIR / "*.parquet"))
    return lf


def extract_surface_features(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Extract surface-layer features for ML training.

    Strategy:
    - "surface" = depth <= 5.0m (top of the water column)
    - "deep" = closest depth to 100m for thermal contrast
    - We group by (time, latitude, longitude) and compute:
        surface_temp, surface_salinity, surface_speed_u, surface_speed_v,
        temp_at_100m, thermal_contrast, current_speed, approx_vorticity

    This is the core feature set for Rakshak cyclone detection.
    """
    print("[FEATURE ENG] Extracting surface features...")

    # Surface layer (depth <= 5m)
    surface = (
        lf.filter(pl.col("depth") <= 5.0)
        .group_by(["time", "latitude", "longitude"])
        .agg([
            pl.col("thetao").mean().alias("sst"),
            pl.col("so").mean().alias("surface_salinity"),
            pl.col("uo").mean().alias("surface_uo"),
            pl.col("vo").mean().alias("surface_vo"),
            pl.col("surface_temp").mean().alias("surface_temp_precomputed"),
        ])
    )

    # Near-100m layer (depth between 80m and 120m)
    deep_100m = (
        lf.filter((pl.col("depth") >= 80.0) & (pl.col("depth") <= 120.0))
        .group_by(["time", "latitude", "longitude"])
        .agg([
            pl.col("thetao").mean().alias("temp_100m"),
        ])
    )

    print("[FEATURE ENG] Collecting surface layer (this may take a moment)...")
    surface_df = surface.collect()
    print(f"  Surface rows: {len(surface_df):,}")

    print("[FEATURE ENG] Collecting 100m layer...")
    deep_df = deep_100m.collect()
    print(f"  100m rows: {len(deep_df):,}")

    # Join surface and 100m on (time, lat, lon)
    print("[FEATURE ENG] Joining layers...")
    df = surface_df.join(
        deep_df,
        on=["time", "latitude", "longitude"],
        how="left",
    )

    # Derived features
    df = df.with_columns([
        # Current speed
        (pl.col("surface_uo") ** 2 + pl.col("surface_vo") ** 2)
        .sqrt()
        .alias("current_speed"),

        # Thermal contrast (warm surface vs cooler 100m = cyclone precursor)
        (pl.col("sst") - pl.col("temp_100m")).alias("thermal_contrast"),

        # Placeholder features (not in current data — will be filled from
        # Copernicus zos download in the next acquisition phase)
        pl.lit(float("nan")).alias("zos"),          # sea surface height anomaly
        pl.lit(float("nan")).alias("mlotst"),       # mixed layer depth
    ])

    # Vorticity: need spatial neighbors, approximate from the dataset itself
    # We sort by lat/lon and compute finite differences across rows.
    # This is an approximation — full vorticity needs a 2D grid.
    # Formula: vorticity ≈ dvo/dlon - duo/dlat
    print("[FEATURE ENG] Computing approximate vorticity...")
    df = df.sort(["time", "latitude", "longitude"])
    df = df.with_columns([
        pl.col("surface_vo").diff().over("time").alias("_dvo"),
        pl.col("surface_uo").diff().over("time").alias("_duo"),
        pl.col("longitude").diff().over("time").alias("_dlon"),
        pl.col("latitude").diff().over("time").alias("_dlat"),
    ])
    df = df.with_columns([
        (
            (pl.col("_dvo") / pl.col("_dlon").clip(1e-6, None)) -
            (pl.col("_duo") / pl.col("_dlat").clip(1e-6, None))
        ).alias("vorticity")
    ]).drop(["_dvo", "_duo", "_dlon", "_dlat"])

    # Select and reorder final feature columns
    feature_cols = [
        "time", "latitude", "longitude",
        "sst", "surface_salinity", "temp_100m", "thermal_contrast",
        "current_speed", "surface_uo", "surface_vo", "vorticity",
        "zos", "mlotst",
    ]
    df = df.select([c for c in feature_cols if c in df.columns])

    print(f"[FEATURE ENG] Feature matrix shape: {df.shape}")
    return df


def save_feature_matrix(df: pl.DataFrame, name: str = "features_unlabeled") -> Path:
    """Save the feature matrix to fabric/ml/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{name}.parquet"
    df.write_parquet(out)
    print(f"[FEATURE ENG] Saved: {out}  ({out.stat().st_size / 1024**2:.1f} MB)")
    return out


if __name__ == "__main__":
    lf = load_parquet_dataset()
    features = extract_surface_features(lf)
    save_feature_matrix(features)
    print("\n[FEATURE ENG] Done. Next step: attach IBTrACS cyclone labels.")
    print("  Run: python services/jal-chakra/plugins/copernicus/label_builder.py")
