import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
train_cyclone_model.py — Rakshak Cyclone Detection Model Training
Leher / Rakshak Intelligence

Trains an XGBoost classifier to detect cyclogenesis conditions.

Input:  fabric/ml/cyclone_training_data.parquet
Output: fabric/ml/anomaly_model.json
        fabric/ml/anomaly_model_metrics.json
        fabric/ml/feature_importance.parquet

Features used:
  sst                — Sea Surface Temperature (°C)
  temp_100m          — Temperature at 100m depth (°C)
  thermal_contrast   — sst - temp_100m (instability indicator)
  current_speed      — sqrt(uo² + vo²) in m/s
  surface_uo         — Eastward current component
  surface_vo         — Northward current component
  vorticity          — Rotational flow indicator
  zos                — Sea Surface Height anomaly (NaN for now)
  mlotst             — Mixed Layer Depth (NaN for now)

Target metric: AUC-ROC > 0.80
"""
import json
import numpy as np
import polars as pl
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

TRAINING_DATA = Path("fabric/ml/cyclone_training_data.parquet")
MODEL_OUT     = Path("fabric/ml/anomaly_model.json")
METRICS_OUT   = Path("fabric/ml/anomaly_model_metrics.json")
IMPORTANCE_OUT = Path("fabric/ml/feature_importance.parquet")

# Features the model will use
# zos and mlotst are included but will be NaN for this training run.
# XGBoost handles NaN natively — it learns the best split direction.
FEATURE_COLS = [
    "sst",
    "temp_100m",
    "thermal_contrast",
    "current_speed",
    "surface_uo",
    "surface_vo",
    "vorticity",
    "zos",       # NaN placeholder — XGBoost handles this natively
    "mlotst",    # NaN placeholder — XGBoost handles this natively
]
LABEL_COL = "label"


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load and prepare the training dataset."""
    print(f"[TRAIN] Loading: {TRAINING_DATA}")
    df = pl.read_parquet(TRAINING_DATA)
    print(f"  Rows: {len(df):,}")
    label_dist = df.group_by("label").len().sort("label")
    for row in label_dist.iter_rows(named=True):
        print(f"    label={row['label']}: {row['len']:,} rows")

    # Keep only rows where core features are non-null
    core_features = ["sst", "current_speed", "thermal_contrast"]
    df = df.filter(
        pl.all_horizontal([pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in core_features])
    )
    print(f"  After dropping core-feature NaNs: {len(df):,}")

    # Fill missing optional features with NaN (XGBoost handles it)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(float("nan")).alias(col))

    X = df.select(FEATURE_COLS).to_numpy(allow_copy=True).astype(np.float32)
    y = df.select(LABEL_COL).to_numpy(allow_copy=True).flatten().astype(np.int8)

    return X, y


def train(X_train, y_train, X_test, y_test) -> xgb.XGBClassifier:
    """Train the XGBoost cyclone detection classifier."""
    # Class weight for imbalanced data: scale_pos_weight = n_neg / n_pos
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[TRAIN] scale_pos_weight = {scale_pos_weight:.2f} (n_neg={n_neg}, n_pos={n_pos})")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        # Enable native NaN handling — no imputation needed
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=20,
    )

    return model


def evaluate(model: xgb.XGBClassifier, X_test, y_test) -> dict:
    """Compute and print evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n[EVAL] AUC-ROC : {auc:.4f}  (target: > 0.80)")
    print(f"[EVAL] F1 Score: {f1:.4f}")
    print(f"[EVAL] Confusion Matrix:\n{cm}")
    print(f"[EVAL] Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    if auc < 0.70:
        print(
            "\n[WARN] AUC < 0.70 — likely using proxy labels.\n"
            "  Acquire historical GLORYS data (1993-2023) + real IBTrACS\n"
            "  tracks to get production-grade AUC-ROC > 0.85."
        )

    return {
        "auc_roc": round(float(auc), 4),
        "f1_score": round(float(f1), 4),
        "n_test_positives": int(y_test.sum()),
        "n_test_negatives": int((y_test == 0).sum()),
        "confusion_matrix": cm.tolist(),
        "feature_cols": FEATURE_COLS,
        "best_iteration": int(model.best_iteration),
    }


def save_feature_importance(model: xgb.XGBClassifier) -> None:
    """Save feature importance to Parquet for inspection."""
    importance = model.get_booster().get_score(importance_type="gain")
    rows = [
        {"feature": feat, "importance_gain": importance.get(f"f{i}", 0.0)}
        for i, feat in enumerate(FEATURE_COLS)
    ]
    df = pl.DataFrame(rows).sort("importance_gain", descending=True)
    df.write_parquet(IMPORTANCE_OUT)
    print(f"\n[TRAIN] Feature importances (by gain):")
    print(df)


if __name__ == "__main__":
    if not TRAINING_DATA.exists():
        print(f"ERROR: {TRAINING_DATA} not found.")
        print("  Run these first:")
        print("  1. python services/jal-chakra/core/feature_engineering.py")
        print("  2. python services/jal-chakra/plugins/copernicus/label_builder.py")
        raise SystemExit(1)

    # Load
    X, y = load_training_data()
    print(f"\n[TRAIN] Feature matrix shape: {X.shape}")

    # Split — temporal split preferred but we don't have time-ordered index here
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[TRAIN] Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

    # Train
    model = train(X_train, y_train, X_test, y_test)

    # Evaluate
    metrics = evaluate(model, X_test, y_test)

    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"\n[TRAIN] Model saved: {MODEL_OUT}")

    # Save metrics
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"[TRAIN] Metrics saved: {METRICS_OUT}")

    # Save feature importance
    save_feature_importance(model)
    print(f"[TRAIN] Feature importance saved: {IMPORTANCE_OUT}")

    print("\n[TRAIN] Cyclone model training complete.")
    print(f"  AUC-ROC: {metrics['auc_roc']}")
    print("  Next step: python services/jal-chakra/core/train_surge_model.py")
