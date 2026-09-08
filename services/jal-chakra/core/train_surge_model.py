"""
train_surge_model.py — Rakshak Storm Surge Regressor Training
Leher / Rakshak Intelligence

Storm surge is separate from cyclone detection.
A cyclone can stay offshore and still push a wall of water onto the coast.

Model: XGBoost Regressor predicting surge height in meters.

Input features (from our parquet data):
  current_speed         — High onshore speed means water is piling up
  sst                   — Warm water = more energy = larger surge
  surface_uo            — Eastward component (for coast-relative direction)
  surface_vo            — Northward component
  coastal_proximity     — We derive this from latitude (proxy for Bay of Bengal)

Missing features (NaN, needed from Copernicus):
  zos                   — Sea Surface Height anomaly (DIRECT surge indicator)
  bathymetry_gradient   — Shallow shelf amplifies surge

Labels:
  Since we have no historical tide gauge surge heights in this dataset,
  we build PROXY labels using the physics:
    surge_proxy = (zos anomaly from SST) approximated as:
    surge_proxy ≈ current_speed * coastal_factor * sst_factor

  When real IMD tide gauge data is available, replace with real heights.

Output: fabric/ml/surge_model.json
"""
import json
import numpy as np
import polars as pl
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

TRAINING_DATA = Path("fabric/ml/cyclone_training_data.parquet")
SURGE_DATA    = Path("fabric/ml/surge_training_data.parquet")
MODEL_OUT     = Path("fabric/ml/surge_model.json")
METRICS_OUT   = Path("fabric/ml/surge_model_metrics.json")

# Coastal latitude bands where surge is amplified
# Bay of Bengal: 8°N–22°N, Arabian Sea: 8°N–23°N (western coast India)
BOB_LAT_MIN, BOB_LAT_MAX = 8.0, 22.0
BOB_LON_MIN, BOB_LON_MAX = 80.0, 100.0

FEATURE_COLS = [
    "sst",
    "current_speed",
    "surface_uo",
    "surface_vo",
    "coastal_proximity",    # derived: 1.0 = within BOB, 0.0 = open ocean
    "zos",                  # NaN placeholder — XGBoost handles natively
]
LABEL_COL = "surge_height_m"  # proxy label


def build_surge_dataset(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build surge training data from the feature matrix by:
    1. Focusing on coastal-adjacent grid cells
    2. Computing a physics-based proxy surge label
    3. Adding coastal proximity feature
    """
    print("[SURGE] Building surge training dataset...")

    # Coastal proximity: cells within Bay of Bengal zone get 1.0, others scaled by distance
    df = df.with_columns([
        pl.when(
            (pl.col("latitude").is_between(BOB_LAT_MIN, BOB_LAT_MAX)) &
            (pl.col("longitude").is_between(BOB_LON_MIN, BOB_LON_MAX))
        )
        .then(pl.lit(1.0))
        .otherwise(
            # Scale by proximity to BOB center
            pl.lit(1.0) - (
                (pl.col("latitude") - 15.0).abs() / 30.0 +
                (pl.col("longitude") - 90.0).abs() / 50.0
            ).clip(0.0, 1.0)
        )
        .alias("coastal_proximity")
    ])

    # Proxy surge label (physics-based approximation):
    #   surge ≈ current_speed * coastal_factor * (sst - 26) / 10
    # This is a training scaffold — replace with real tide gauge heights later.
    df = df.with_columns([
        (
            pl.col("current_speed") *
            pl.col("coastal_proximity") *
            ((pl.col("sst") - 26.0) / 10.0).clip(0.0, 2.0) *
            2.0  # scale to plausible surge heights in meters
        )
        .clip(0.0, 5.0)  # cap at 5m (extreme but physically possible)
        .alias(LABEL_COL)
    ])

    # Keep only coastal-adjacent cells (proximity > 0.3)
    df = df.filter(pl.col("coastal_proximity") > 0.3)
    print(f"  Coastal cells: {len(df):,}")

    return df.select(FEATURE_COLS + [LABEL_COL, "latitude", "longitude"])


def train_surge_model(X_train, y_train, X_test, y_test) -> xgb.XGBRegressor:
    """Train the XGBoost storm surge regressor."""
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=20,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=20,
    )
    return model


if __name__ == "__main__":
    if not TRAINING_DATA.exists():
        print(f"ERROR: {TRAINING_DATA} not found. Run feature_engineering.py and label_builder.py first.")
        raise SystemExit(1)

    # Load and filter to cells that have the needed columns
    print(f"[SURGE] Loading: {TRAINING_DATA}")
    df = pl.read_parquet(TRAINING_DATA)
    print(f"  Rows loaded: {len(df):,}")

    # Check for required columns
    for col in ["sst", "current_speed", "surface_uo", "surface_vo", "latitude", "longitude"]:
        if col not in df.columns:
            print(f"ERROR: Column '{col}' not found. Feature engineering may be incomplete.")
            raise SystemExit(1)

    # Add zos as NaN if missing
    if "zos" not in df.columns:
        df = df.with_columns(pl.lit(float("nan")).alias("zos"))

    # Build surge dataset
    surge_df = build_surge_dataset(df)

    # Save the surge training set
    SURGE_DATA.parent.mkdir(parents=True, exist_ok=True)
    surge_df.write_parquet(SURGE_DATA)
    print(f"[SURGE] Saved surge training data: {SURGE_DATA}")

    # Prepare arrays — XGBoost handles NaN natively
    X = surge_df.select(FEATURE_COLS).to_numpy(allow_copy=True).astype(np.float32)
    y = surge_df.select(LABEL_COL).to_numpy(allow_copy=True).flatten().astype(np.float32)

    print(f"[SURGE] Feature matrix: {X.shape}, Label range: [{y.min():.3f}, {y.max():.3f}]")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = train_surge_model(X_train, y_train, X_test, y_test)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n[SURGE] MAE  : {mae:.4f} m  (mean absolute error in surge height)")
    print(f"[SURGE] R²   : {r2:.4f}")

    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"[SURGE] Model saved: {MODEL_OUT}")

    metrics = {
        "mae_meters": round(float(mae), 4),
        "r2_score": round(float(r2), 4),
        "n_test": len(y_test),
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "note": "Proxy labels used. Replace with IMD tide gauge heights for production.",
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"[SURGE] Metrics saved: {METRICS_OUT}")
    print("\n[SURGE] Storm surge model training complete.")
    print("  Next step: python services/jal-chakra/core/rakshak.py")
