"""
label_builder.py — IBTrACS Cyclone Label Joiner
Leher / Jal Chakra Pipeline / Copernicus Plugin

Downloads IBTrACS Indian Ocean cyclone track CSV and joins it to the
feature matrix to create a labeled training dataset for Rakshak.

Label logic:
  - A grid cell is POSITIVE (label=1) if a cyclone track point falls
    within MATCH_RADIUS_DEG degrees of that cell on the same day.
  - All other ocean cells are NEGATIVE (label=0).
  - We balance to ~1:5 positive:negative ratio.

IBTrACS source: North Indian Ocean basin (NI)
URL: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv
"""
import urllib.request
import numpy as np
import polars as pl
from pathlib import Path
from datetime import timedelta

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)
IBTRACS_PATH = Path("fabric/ml/ibtracs_indian_ocean.csv")
FEATURE_PATH = Path("fabric/ml/features_unlabeled.parquet")
OUTPUT_PATH  = Path("fabric/ml/cyclone_training_data.parquet")

# A grid cell is "near a cyclone" if within this radius (degrees ~= 111km/deg)
MATCH_RADIUS_DEG = 2.5  # ~278 km — generous but avoids false negatives


def download_ibtracs() -> Path:
    """Download IBTrACS NI (North Indian Ocean) CSV if not already present."""
    IBTRACS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if IBTRACS_PATH.exists():
        print(f"[IBTrACS] Already downloaded: {IBTRACS_PATH}")
        return IBTRACS_PATH

    print("[IBTrACS] Downloading North Indian Ocean tracks...")
    print(f"  URL: {IBTRACS_URL}")
    urllib.request.urlretrieve(IBTRACS_URL, IBTRACS_PATH)
    size_mb = IBTRACS_PATH.stat().st_size / 1024**2
    print(f"  Saved: {IBTRACS_PATH} ({size_mb:.1f} MB)")
    return IBTRACS_PATH


def load_ibtracs(path: Path) -> pl.DataFrame:
    """
    Load IBTrACS CSV, properly handling its two-row header format:
      Row 0: real column names  (SID, SEASON, ..., ISO_TIME, ..., LAT, LON, ...)
      Row 1: unit descriptors   (Year, , , , , degrees_north, degrees_east, kts, ...)
      Row 2+: actual data

    Strategy: read real column names from row 0 manually, then load
    data starting at row 2 with has_header=False and assign those names.
    """
    print("[IBTrACS] Parsing tracks...")

    # Step 1: Read the real header from row 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
    real_cols = [c.strip() for c in header_line.split(",")]
    print(f"  Real columns (first 12): {real_cols[:12]}")

    # Step 2: Read data rows, skipping both header row AND unit-descriptor row
    df = pl.read_csv(
        path,
        skip_rows=2,          # skip row 0 (header) and row 1 (unit descriptors)
        has_header=False,     # we will assign column names manually
        new_columns=real_cols,
        null_values=["", " "],
        infer_schema_length=200,
        ignore_errors=True,
    )
    print(f"  Raw rows: {len(df):,}")

    # Step 3: Map to canonical names  (IBTrACS uses uppercase)
    col_map = {}
    for col in df.columns:
        cu = col.strip().upper()
        if cu == "ISO_TIME":
            col_map["time"] = col
        elif cu == "LAT":
            col_map["latitude"] = col
        elif cu == "LON":
            col_map["longitude"] = col

    missing = [k for k in ("time", "latitude", "longitude") if k not in col_map]
    if missing:
        raise ValueError(
            f"IBTrACS CSV is missing expected columns: {missing}\n"
            f"Available columns (first 20): {df.columns[:20]}"
        )

    df = df.rename({v: k for k, v in col_map.items()})

    # Step 4: Cast lat/lon to float32
    df = df.with_columns([
        pl.col("latitude").cast(pl.Float32, strict=False),
        pl.col("longitude").cast(pl.Float32, strict=False),
    ])

    # Step 5: Drop rows with missing position
    df = df.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())

    # Step 6: Extract date (YYYY-MM-DD) from ISO_TIME for day-level matching
    df = df.with_columns([
        pl.col("time").str.slice(0, 10).alias("date")
    ])

    print(f"  Clean track rows: {len(df):,}")
    return df.select(["date", "latitude", "longitude"])


def attach_labels(features: pl.DataFrame, tracks: pl.DataFrame) -> pl.DataFrame:
    """
    Attach binary cyclone labels to the feature matrix.

    For each unique date in the feature matrix, check which grid cells
    fall within MATCH_RADIUS_DEG of any track point on that date.

    Returns the feature matrix with a new 'label' column (0 or 1).
    """
    print("[LABEL] Attaching cyclone labels...")

    # Precompute date from time column in features
    features = features.with_columns([
        pl.col("time").dt.strftime("%Y-%m-%d").alias("date")
    ])

    # Unique dates in features
    feature_dates = features.select("date").unique().to_series().to_list()
    track_dates = tracks.select("date").unique().to_series().to_list()

    overlapping = set(feature_dates) & set(track_dates)
    print(f"  Feature dates: {len(feature_dates)}, track dates: {len(track_dates)}")
    print(f"  Overlapping dates: {len(overlapping)}")

    if not overlapping:
        print(
            "[LABEL] WARNING: No date overlap between features and IBTrACS tracks.\n"
            "  The feature data is from Aug-Sep 2026, but IBTrACS may not have\n"
            "  that recent data yet. We will use SPATIAL proximity to known\n"
            "  cyclone-prone regions as a proxy label for now.\n"
            "  REAL labels will come once you download historical GLORYS + track data."
        )
        return _apply_proxy_labels(features)

    # For each date with track data, mark nearby cells
    label_rows = []

    for date in overlapping:
        day_tracks = tracks.filter(pl.col("date") == date)
        day_feats = features.filter(pl.col("date") == date)

        track_lats = day_tracks["latitude"].to_numpy()
        track_lons = day_tracks["longitude"].to_numpy()

        feat_lats = day_feats["latitude"].to_numpy()
        feat_lons = day_feats["longitude"].to_numpy()

        # Vectorized distance: min distance to any track point
        # Shape: (n_cells, n_track_points)
        dlat = feat_lats[:, None] - track_lats[None, :]
        dlon = feat_lons[:, None] - track_lons[None, :]
        dist = np.sqrt(dlat**2 + dlon**2)
        min_dist = dist.min(axis=1)

        labels = (min_dist <= MATCH_RADIUS_DEG).astype(np.int8)
        label_rows.append(
            day_feats.with_columns(pl.Series("label", labels))
        )

    # Non-overlapping dates get label=0
    non_overlap_feats = features.filter(~pl.col("date").is_in(list(overlapping)))
    if len(non_overlap_feats) > 0:
        non_overlap_feats = non_overlap_feats.with_columns(pl.lit(0).cast(pl.Int8).alias("label"))
        label_rows.append(non_overlap_feats)

    result = pl.concat(label_rows)

    pos = result.filter(pl.col("label") == 1)
    neg = result.filter(pl.col("label") == 0)
    print(f"  Positive (cyclone) samples: {len(pos):,}")
    print(f"  Negative (non-cyclone) samples: {len(neg):,}")
    print(f"  Positive rate: {100 * len(pos) / len(result):.2f}%")

    return result.drop("date")


def _apply_proxy_labels(features: pl.DataFrame) -> pl.DataFrame:
    """
    When there is no date overlap with IBTrACS, apply proxy labels based on
    oceanographic conditions typical of cyclone environments.

    This is NOT a substitute for real labels — it demonstrates the pipeline
    and allows model training to proceed while historical data is acquired.

    Proxy positive criteria (all must be met):
    - SST > 28°C (warm ocean, necessary but not sufficient)
    - Current speed > 0.4 m/s
    - Thermal contrast > 5°C (warm surface, cooler below = instability)
    """
    print("[LABEL] Applying proxy labels based on oceanographic thresholds...")

    features = features.with_columns([
        pl.when(
            (pl.col("sst") > 28.0) &
            (pl.col("current_speed") > 0.4) &
            (pl.col("thermal_contrast") > 5.0)
        )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
        .alias("label")
    ])

    pos = features.filter(pl.col("label") == 1)
    neg = features.filter(pl.col("label") == 0)
    print(f"  Proxy positive: {len(pos):,}")
    print(f"  Proxy negative: {len(neg):,}")
    print(f"  NOTE: These are proxy labels. Train on real historical data when available.")

    return features


def balance_dataset(df: pl.DataFrame, ratio: int = 5) -> pl.DataFrame:
    """
    Balance the dataset to ratio:1 negative:positive.
    Keeps all positives, samples negatives down to ratio * n_positives.
    """
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)

    n_pos = len(pos)
    n_neg_target = min(len(neg), ratio * n_pos)

    if len(neg) > n_neg_target:
        neg = neg.sample(n=n_neg_target, seed=42)

    balanced = pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=42)
    print(f"[LABEL] Balanced dataset: {len(pos):,} pos + {len(neg):,} neg = {len(balanced):,} total")
    return balanced


if __name__ == "__main__":
    # Step 1: Download labels
    ibtracs_path = download_ibtracs()

    # Step 2: Load tracks
    tracks = load_ibtracs(ibtracs_path)

    # Step 3: Load feature matrix
    print(f"\n[LABEL] Loading feature matrix from {FEATURE_PATH}...")
    if not FEATURE_PATH.exists():
        print(f"ERROR: {FEATURE_PATH} not found. Run feature_engineering.py first.")
        raise SystemExit(1)
    features = pl.read_parquet(FEATURE_PATH)
    print(f"  Feature rows: {len(features):,}")

    # Step 4: Attach labels
    labeled = attach_labels(features, tracks)

    # Step 5: Balance
    balanced = balance_dataset(labeled)

    # Step 6: Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    balanced.write_parquet(OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / 1024**2
    print(f"\n[LABEL] Saved labeled training data: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("  Next step: python services/jal-chakra/core/train_cyclone_model.py")
