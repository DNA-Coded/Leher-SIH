# Leher — Your Execution Plan: Data Pipeline + ML Model

**Your role:** Data Engineer / ML Engineer
**Your deliverables:** Processed Zarr datasets + trained ML models + handover package
**Source of truth:** [LeherGuide.md](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md)

---

## Phase 1: Project Setup & First Data Acquisition (Day 1)

> **Goal:** Create the folder structure, install dependencies, download your first real Indian Ocean NetCDF file.
> **LeherGuide reference:** Sections [5](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L192) (Repo Structure), [6](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L312) (Data Sources), [7.2](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L398) (Utpatti)

### Step 1.1 — Create the Monorepo Skeleton

Create all the directories that your pipeline will write into:

```
Leher/
├── services/
│   └── jal-chakra/           ← Your pipeline code lives here
│       ├── core/
│       │   ├── pipeline.py
│       │   ├── validator.py
│       │   └── normalizer.py
│       └── plugins/
│           ├── copernicus/
│           └── argo/
├── fabric/                   ← Samudra Data Fabric (your output)
│   ├── raw/
│   │   ├── copernicus/glorys/
│   │   └── argo/profiles/
│   ├── quarantined/
│   ├── datasets/
│   │   ├── glorys/
│   │   └── argo/
│   └── ml/                   ← Your trained models go here
├── testdata/
│   └── golden/
├── notebooks/                ← Your exploration notebooks
└── LeherGuide.md
```

### Step 1.2 — Install Python Dependencies

```bash
pip install copernicusmarine xarray zarr[v3] numpy polars duckdb pyarrow xgboost scikit-learn matplotlib jupyter netCDF4 blosc2
```

### Step 1.3 — Create Copernicus Account

1. Go to https://data.marine.copernicus.eu/register
2. Create a free account
3. Run `copernicusmarine login` to save credentials locally

### Step 1.4 — Download Your First NetCDF (Utpatti Stage)

This is the most exciting step — you are pulling real Indian Ocean data from the Copernicus satellite-validated model.

```python
# services/jal-chakra/plugins/copernicus/download.py
import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    variables=["thetao", "so", "uo", "vo", "zos", "mlotst", "bottomT", "siconc", "sithick"],
    minimum_longitude=20.0,
    maximum_longitude=130.0,
    minimum_latitude=-40.0,
    maximum_latitude=30.0,
    minimum_depth=0.0,
    maximum_depth=5500.0,
    start_datetime="2024-01-01T00:00:00",
    end_datetime="2024-01-01T23:59:59",
    output_filename="raw_glorys_20240101.nc",
    output_directory="../../fabric/raw/copernicus/glorys/"
)
```

### Step 1.5 — Verify the Download

```python
# notebooks/01_verify_download.ipynb
import xarray as xr

ds = xr.open_dataset("fabric/raw/copernicus/glorys/raw_glorys_20240101.nc")
print(ds)                    # Should show thetao, so, uo, vo, zos, mlotst, bottomT, siconc, sithick
print(ds.dims)               # Should show time, depth, latitude, longitude
print(ds.thetao.attrs)       # Should show units: degrees_C
print(ds.depth.values[:10])  # First 10 native depth levels
print(f"Grid shape: {ds.thetao.shape}")  # Should be ~(1, 50, 840, 1320)
```

> **Success criteria:** You see a ~400MB NetCDF file in `fabric/raw/` with 9 variables, 50 depth levels, covering the Indian Ocean BBOX.

---

## Phase 2: Jal Chakra Pipeline — Validate & Convert (Day 2)

> **Goal:** Build the Pariksha (validation) and Rupantar (transformation) stages. Convert raw NetCDF to chunked Zarr.
> **LeherGuide reference:** Sections [7.3](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L422) (Pariksha), [7.4](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L451) (Rupantar), [9](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L557) (Samudra Fabric)

### Step 2.1 — Build the Validator (Pariksha)

```python
# services/jal-chakra/core/validator.py
import xarray as xr
from pathlib import Path
import json

class ValidationError(Exception):
    pass

def validate_glorys(nc_path: Path) -> xr.Dataset:
    """Pariksha: Validate a raw GLORYS NetCDF file before processing."""
    ds = xr.open_dataset(nc_path)
    errors = []

    # 1. Check required variables exist
    required = {"thetao", "so", "uo", "vo", "zos", "mlotst", "bottomT", "siconc", "sithick"}
    missing = required - set(ds.data_vars)
    if missing:
        errors.append(f"Missing variables: {missing}")

    # 2. Check coordinate ranges
    if ds.latitude.min() < -90 or ds.latitude.max() > 90:
        errors.append(f"Latitude out of range: [{ds.latitude.min()}, {ds.latitude.max()}]")

    if ds.longitude.min() < -180 or ds.longitude.max() > 360:
        errors.append(f"Longitude out of range: [{ds.longitude.min()}, {ds.longitude.max()}]")

    # 3. Check depth is positive (positive-down convention)
    if "depth" in ds.coords:
        if float(ds.depth.min()) < 0:
            errors.append("Depth values are negative — expected positive-down convention")
    else:
        errors.append("No 'depth' coordinate found")

    # 4. Check units
    if hasattr(ds.thetao, "units"):
        if ds.thetao.units not in ["degrees_C", "degC", "degree_Celsius"]:
            errors.append(f"Temperature units unexpected: {ds.thetao.units}")

    # 5. Check for fill_value != 0
    for var in required:
        if var in ds.data_vars:
            fv = ds[var].encoding.get("_FillValue", None)
            if fv == 0.0:
                errors.append(f"{var} has _FillValue=0.0 — this is dangerous, NaN will become 0")

    if errors:
        # Move to quarantine
        error_record = {"file": str(nc_path), "errors": errors}
        quarantine_log = Path("fabric/quarantined/errors.jsonl")
        quarantine_log.parent.mkdir(parents=True, exist_ok=True)
        with open(quarantine_log, "a") as f:
            f.write(json.dumps(error_record) + "\n")
        raise ValidationError(f"Validation failed: {errors}")

    print(f"✓ Pariksha PASSED: {nc_path.name}")
    return ds
```

### Step 2.2 — Build the Converter (Rupantar)

```python
# services/jal-chakra/core/normalizer.py
import xarray as xr
import numpy as np
from pathlib import Path
import json
import hashlib

def convert_to_zarr(ds: xr.Dataset, source_name: str, revision: str) -> Path:
    """Rupantar: Convert validated NetCDF to chunked Zarr in Samudra Data Fabric."""

    output_base = Path(f"fabric/datasets/{source_name}/{revision}")
    output_base.mkdir(parents=True, exist_ok=True)

    # Canonical variable mapping
    var_map = {
        "thetao": "temperature",
        "so": "salinity",
        "uo": "u_velocity",
        "vo": "v_velocity",
        "zos": "sea_surface_height",
        "mlotst": "mixed_layer_depth",
        "bottomT": "sea_floor_temperature",
        "siconc": "sea_ice_concentration",
        "sithick": "sea_ice_thickness",
        "chl": "chlorophyll_a"
    }

    manifest_vars = []

    for src_name, canon_name in var_map.items():
        if src_name not in ds.data_vars:
            continue

        da = ds[src_name]

        # Ensure float32 and NaN fill (NEVER zero)
        data = da.values.astype(np.float32)
        # Replace any masked/fill values with NaN
        data = np.where(np.isfinite(data), data, np.float32("nan"))

        # Rebuild as a clean DataArray
        clean_da = xr.DataArray(
            data,
            dims=da.dims,
            coords=da.coords,
            name=canon_name,
            attrs=da.attrs,
        )

        zarr_path = output_base / f"{canon_name}.zarr"

        # Chunk shape: (1 time, 1 depth, full lat, full lon)
        chunks = {}
        if "time" in clean_da.dims:
            chunks["time"] = 1
        if "depth" in clean_da.dims:
            chunks["depth"] = 1

        clean_ds = clean_da.to_dataset()
        clean_ds.to_zarr(
            str(zarr_path),
            mode="w",
            consolidated=True,
            encoding={
                canon_name: {
                    "chunks": tuple(chunks.get(d, clean_da.sizes[d]) for d in clean_da.dims),
                    "compressor": None,  # Use default blosc
                    "dtype": "float32",
                }
            }
        )

        data_finite = data[np.isfinite(data)]
        manifest_vars.append({
            "canonical_name": canon_name,
            "source_name": src_name,
            "units": str(da.attrs.get("units", "unknown")),
            "standard_name": str(da.attrs.get("standard_name", "")),
            "data_min": float(np.nanmin(data_finite)) if len(data_finite) > 0 else None,
            "data_max": float(np.nanmax(data_finite)) if len(data_finite) > 0 else None,
            "zarr_path": str(zarr_path.relative_to("fabric")),
        })

        print(f"  ✓ {src_name} → {canon_name}.zarr ({zarr_path})")

    # Compute derived product: current_speed = sqrt(uo² + vo²)
    if "uo" in ds.data_vars and "vo" in ds.data_vars:
        uo = ds["uo"].values.astype(np.float32)
        vo = ds["vo"].values.astype(np.float32)
        speed = np.sqrt(uo**2 + vo**2)

        speed_da = xr.DataArray(
            speed, dims=ds["uo"].dims, coords=ds["uo"].coords, name="current_speed"
        )
        speed_path = output_base / "current_speed.zarr"
        speed_da.to_dataset().to_zarr(str(speed_path), mode="w", consolidated=True)

        speed_finite = speed[np.isfinite(speed)]
        manifest_vars.append({
            "canonical_name": "current_speed",
            "source_name": "derived(uo, vo)",
            "units": "m s-1",
            "standard_name": "sea_water_speed",
            "data_min": float(np.nanmin(speed_finite)) if len(speed_finite) > 0 else None,
            "data_max": float(np.nanmax(speed_finite)) if len(speed_finite) > 0 else None,
            "zarr_path": str(speed_path.relative_to("fabric")),
        })
        print(f"  ✓ derived → current_speed.zarr")

    return output_base, manifest_vars
```

### Step 2.3 — Build the Publisher (Prakashan)

```python
# services/jal-chakra/core/pipeline.py
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from .validator import validate_glorys
from .normalizer import convert_to_zarr

def compute_checksum(directory: Path) -> str:
    """SHA-256 over all files in the Zarr directory."""
    h = hashlib.sha256()
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()

def run_jal_chakra(nc_path: Path, source_name: str = "glorys", revision: str = None):
    """Run the full 4-stage Jal Chakra pipeline."""

    if revision is None:
        revision = f"v{nc_path.stem.split('_')[-1]}"  # e.g., v20240101

    print(f"═══ JAL CHAKRA PIPELINE ═══")
    print(f"Source: {nc_path}")
    print(f"Revision: {revision}")
    print()

    # Stage 1: Utpatti (already done — file exists on disk)
    print("▸ UTPATTI (Origin): File acquired")

    # Stage 2: Pariksha (Examination)
    print("▸ PARIKSHA (Examination): Validating...")
    ds = validate_glorys(nc_path)

    # Stage 3: Rupantar (Transformation)
    print("▸ RUPANTAR (Transformation): Converting to Zarr...")
    output_base, manifest_vars = convert_to_zarr(ds, source_name, revision)

    # Stage 4: Prakashan (Illumination)
    print("▸ PRAKASHAN (Illumination): Publishing manifest...")

    # Deactivate any previous version
    datasets_dir = Path("fabric/datasets") / source_name
    for old_manifest in datasets_dir.glob("*/manifest.json"):
        old = json.loads(old_manifest.read_text())
        if old.get("is_active"):
            old["is_active"] = False
            old_manifest.write_text(json.dumps(old, indent=2))
            print(f"  ⤷ Deactivated previous: {old['revision_label']}")

    manifest = {
        "dataset_id": f"{source_name}_{revision}",
        "source_id": "GLOBAL_MULTIYEAR_PHY_001_030",
        "revision_label": revision,
        "is_active": True,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "time_min": str(ds.time.values[0]) if "time" in ds.coords else None,
        "time_max": str(ds.time.values[-1]) if "time" in ds.coords else None,
        "depth_min_m": float(ds.depth.min()) if "depth" in ds.coords else 0,
        "depth_max_m": float(ds.depth.max()) if "depth" in ds.coords else 0,
        "native_depths": [float(d) for d in ds.depth.values] if "depth" in ds.coords else [],
        "bbox": {
            "lon_min": float(ds.longitude.min()),
            "lon_max": float(ds.longitude.max()),
            "lat_min": float(ds.latitude.min()),
            "lat_max": float(ds.latitude.max()),
        },
        "variables": manifest_vars,
        "provenance": {
            "source_url": "https://data.marine.copernicus.eu/",
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "sha256": compute_checksum(output_base),
            "cf_validation": "PASS",
        }
    }

    manifest_path = output_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  ✓ Manifest written: {manifest_path}")
    print()
    print(f"═══ PIPELINE COMPLETE ═══")
    print(f"Active dataset: {manifest['dataset_id']}")
    print(f"Variables: {[v['canonical_name'] for v in manifest_vars]}")
    print(f"Depths: {len(manifest['native_depths'])} levels")

    return manifest
```

### Step 2.4 — Run the Full Pipeline

```python
# run_pipeline.py (at Leher root)
from pathlib import Path
from services.jal_chakra.core.pipeline import run_jal_chakra

nc_file = Path("fabric/raw/copernicus/glorys/raw_glorys_20240101.nc")
manifest = run_jal_chakra(nc_file)
```

> **Success criteria:**
> - `fabric/datasets/glorys/v20240101/temperature.zarr/` exists
> - `fabric/datasets/glorys/v20240101/manifest.json` exists and has `is_active: true`
> - `current_speed.zarr` derived product is present
> - No NaN values became 0.0 (verify with `xr.open_zarr(...).temperature.isel(time=0, depth=0).values`)

---

## Phase 3: Argo Instrument Data Ingestion (Day 3)

> **Goal:** Download and process Argo float profiles for the Indian Ocean.
> **LeherGuide reference:** Section [6.3](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L358) (Argo GDAC)

### Step 3.1 — Download Argo Profiles

```python
# services/jal-chakra/plugins/argo/download.py
import urllib.request
from pathlib import Path

ARGO_GDAC = "https://data-argo.ifremer.fr/geo/indian_ocean"

def download_argo_index(output_dir: Path):
    """Download the Argo profile index for the Indian Ocean."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_url = "https://data-argo.ifremer.fr/ar_index_global_prof.txt"
    target = output_dir / "ar_index_global_prof.txt"

    if not target.exists():
        print("Downloading Argo index (~50MB)...")
        urllib.request.urlretrieve(index_url, target)
        print(f"✓ Saved to {target}")

    return target
```

### Step 3.2 — Parse and Filter Indian Ocean Profiles

```python
# services/jal-chakra/plugins/argo/process.py
import polars as pl
from pathlib import Path

def filter_indian_ocean_profiles(index_path: Path) -> pl.DataFrame:
    """Filter Argo index for Indian Ocean BBOX, QC=1 only."""

    # Read the Argo index (CSV with # comments)
    df = pl.read_csv(
        index_path,
        comment_prefix="#",
        separator=",",
        has_header=True,
        null_values=[""],
    )

    # Filter to Indian Ocean BBOX: 20°E-130°E, 40°S-30°N
    indian_ocean = df.filter(
        (pl.col("latitude") >= -40) & (pl.col("latitude") <= 30) &
        (pl.col("longitude") >= 20) & (pl.col("longitude") <= 130)
    )

    print(f"Total profiles: {len(df)}")
    print(f"Indian Ocean profiles: {len(indian_ocean)}")

    return indian_ocean
```

### Step 3.3 — Convert to Parquet for DuckDB

```python
# Save as Parquet in the Samudra Data Fabric
def save_argo_profiles(profiles: pl.DataFrame, revision: str = "v20240115"):
    output_dir = Path(f"fabric/datasets/argo/{revision}")
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "profiles.parquet"
    profiles.write_parquet(parquet_path)

    # Write manifest
    manifest = {
        "dataset_id": f"argo_{revision}",
        "source_id": "ARGO_GDAC",
        "revision_label": revision,
        "is_active": True,
        "profile_count": len(profiles),
        "bbox": {
            "lon_min": 20.0, "lon_max": 130.0,
            "lat_min": -40.0, "lat_max": 30.0,
        },
    }
    import json
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✓ Saved {len(profiles)} profiles to {parquet_path}")
```

> **Success criteria:** `fabric/datasets/argo/v20240115/profiles.parquet` exists with Indian Ocean Argo profiles filtered by BBOX.

---

## Phase 4: DuckDB Catalog — Wire Everything Together (Day 3-4)

> **Goal:** Build the DuckDB catalog that reads all manifests and makes them queryable.
> **LeherGuide reference:** Section [10](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L604) (DuckDB Embedded Catalog)

### Step 4.1 — Build the Catalog Loader

```python
# services/jal-chakra/core/catalog.py
import duckdb
from pathlib import Path

def init_catalog(fabric_root: Path = Path("fabric")) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB in-memory catalog from Samudra Data Fabric manifests."""
    con = duckdb.connect(":memory:")

    # Install and load spatial extension for geo queries
    con.execute("INSTALL spatial; LOAD spatial;")

    # Load all dataset manifests
    manifest_pattern = str(fabric_root / "datasets" / "*" / "*" / "manifest.json")
    con.execute(f"""
        CREATE TABLE catalog AS
        SELECT * FROM read_json_auto('{manifest_pattern}', format='auto', filename=true)
    """)

    # Load Argo profiles if they exist
    argo_pattern = str(fabric_root / "datasets" / "argo" / "*" / "profiles.parquet")
    try:
        con.execute(f"""
            CREATE TABLE instrument_profiles AS
            SELECT * FROM read_parquet('{argo_pattern}')
        """)
        argo_count = con.execute("SELECT COUNT(*) FROM instrument_profiles").fetchone()[0]
        print(f"✓ Loaded {argo_count} Argo profiles")
    except Exception:
        print("⚠ No Argo profiles found yet")

    catalog_count = con.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
    print(f"✓ DuckDB catalog ready: {catalog_count} dataset(s)")

    return con
```

### Step 4.2 — Test Catalog Queries

```python
# notebooks/02_test_catalog.ipynb
from services.jal_chakra.core.catalog import init_catalog

con = init_catalog()

# Find active datasets
print(con.execute("""
    SELECT dataset_id, revision_label, is_active
    FROM catalog
    WHERE is_active = true
""").fetchdf())

# Find temperature Zarr path
print(con.execute("""
    SELECT unnest(variables) as var
    FROM catalog
    WHERE dataset_id LIKE 'glorys%' AND is_active = true
""").fetchdf())
```

> **Success criteria:** DuckDB loads manifests in microseconds. SQL queries return your GLORYS and Argo datasets correctly.

---

## Phase 5: ML Model — Cyclone Detection (Days 4-6)

> **Goal:** Train the XGBoost cyclone detection model using historical data.
> **LeherGuide reference:** Sections [21.1](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L1441) (Cyclone Model), [21.4](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L1500) (Training Data Acquisition)

### Step 5.1 — Download Historical Cyclone Tracks

```python
# services/jal-chakra/plugins/copernicus/training_data.py
import polars as pl
import urllib.request
from pathlib import Path

def download_ibtracs():
    """Download IBTrACS Indian Ocean cyclone tracks (CSV)."""
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
    target = Path("fabric/ml/ibtracs_indian_ocean.csv")
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        print("Downloading IBTrACS Indian Ocean tracks...")
        urllib.request.urlretrieve(url, target)

    df = pl.read_csv(target, skip_rows=1, null_values=["", " "])
    print(f"✓ {len(df)} cyclone track records loaded")
    return df
```

### Step 5.2 — Feature Engineering

```python
# notebooks/03_cyclone_feature_engineering.ipynb
import xarray as xr
import numpy as np
from pathlib import Path

def extract_features_for_point(zarr_base: Path, lat: float, lon: float) -> dict:
    """Extract ML features for a single grid cell from Zarr data."""

    temp_ds = xr.open_zarr(str(zarr_base / "temperature.zarr"))
    speed_ds = xr.open_zarr(str(zarr_base / "current_speed.zarr"))
    ssh_ds = xr.open_zarr(str(zarr_base / "sea_surface_height.zarr"))

    # Surface temperature
    sst = float(temp_ds.temperature.sel(
        depth=0.5, latitude=lat, longitude=lon, method="nearest"
    ).isel(time=0).values)

    # Temperature at 100m
    t100 = float(temp_ds.temperature.sel(
        depth=100, latitude=lat, longitude=lon, method="nearest"
    ).isel(time=0).values)

    # Surface current speed
    spd = float(speed_ds.current_speed.sel(
        depth=0.5, latitude=lat, longitude=lon, method="nearest"
    ).isel(time=0).values)

    # Sea surface height anomaly
    zos = float(ssh_ds.sea_surface_height.sel(
        latitude=lat, longitude=lon, method="nearest"
    ).isel(time=0).values)

    # Current vorticity (finite differences around the point)
    uo_ds = xr.open_zarr(str(zarr_base / "u_velocity.zarr"))
    vo_ds = xr.open_zarr(str(zarr_base / "v_velocity.zarr"))

    uo_slice = uo_ds.u_velocity.sel(
        depth=0.5, method="nearest"
    ).isel(time=0).sel(
        latitude=slice(lat - 1, lat + 1),
        longitude=slice(lon - 1, lon + 1)
    ).values

    vo_slice = vo_ds.v_velocity.sel(
        depth=0.5, method="nearest"
    ).isel(time=0).sel(
        latitude=slice(lat - 1, lat + 1),
        longitude=slice(lon - 1, lon + 1)
    ).values

    # Vorticity = dv/dx - du/dy (finite difference)
    dvdx = np.gradient(vo_slice, axis=1)
    dudy = np.gradient(uo_slice, axis=0)
    vorticity = float(np.nanmean(dvdx - dudy))

    return {
        "sst": sst,
        "t100": t100,
        "thermal_contrast": sst - t100,
        "current_speed": spd,
        "vorticity": vorticity,
        "zos": zos,
    }
```

### Step 5.3 — Build Training Dataset

```python
# notebooks/03_cyclone_feature_engineering.ipynb (continued)
import polars as pl

def build_training_dataset(cyclone_tracks: pl.DataFrame, zarr_base: Path):
    """Build positive + negative samples for cyclone detection."""
    positive_samples = []
    negative_samples = []

    for row in cyclone_tracks.iter_rows(named=True):
        lat, lon = row["LAT"], row["LON"]

        try:
            features = extract_features_for_point(zarr_base, lat, lon)
            features["label"] = 1  # Positive: cyclone present
            positive_samples.append(features)
        except Exception:
            continue

        # Generate 5 negative samples at random locations
        for _ in range(5):
            rand_lat = np.random.uniform(-30, 25)
            rand_lon = np.random.uniform(40, 120)
            try:
                neg_features = extract_features_for_point(zarr_base, rand_lat, rand_lon)
                neg_features["label"] = 0
                negative_samples.append(neg_features)
            except Exception:
                continue

    all_samples = positive_samples + negative_samples
    return pl.DataFrame(all_samples)
```

### Step 5.4 — Train the XGBoost Model

```python
# notebooks/04_train_cyclone_model.ipynb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# Load your built training dataset
training_data = pl.read_parquet("fabric/ml/cyclone_training_data.parquet")

feature_cols = ["sst", "t100", "thermal_contrast", "current_speed", "vorticity", "zos"]
X = training_data[feature_cols].to_numpy()
y = training_data["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="auc",
    early_stopping_rounds=20,
    use_label_encoder=False,
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=20)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred))

# Save to Samudra Data Fabric
model.save_model("fabric/ml/anomaly_model.json")
print("✓ Model saved to fabric/ml/anomaly_model.json")
```

> **Success criteria:** AUC-ROC > 0.80 on the test set. Model saved to `fabric/ml/anomaly_model.json`.

---

## Phase 6: ML Model — Storm Surge & Safe Zones (Days 6-7)

> **Goal:** Build the storm surge model and the safe zone classification algorithm.
> **LeherGuide reference:** Sections [21.3](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L1482) (Storm Surge), [22](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L1582) (Safe Zones)

### Step 6.1 — Storm Surge Regressor

```python
# notebooks/05_train_surge_model.ipynb
import xgboost as xgb

surge_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    objective="reg:squarederror",
)

# Features: zos, current_speed, current_direction_coastal, bathymetry_gradient, distance_to_coast
# Labels: historical surge heights from IMD tide gauge records
surge_model.fit(X_surge_train, y_surge_heights)
surge_model.save_model("fabric/ml/surge_model.json")
print("✓ Surge model saved")
```

### Step 6.2 — Safe Zone Classification Function

```python
# services/jal-chakra/core/safe_zones.py
import numpy as np
import xarray as xr
from pathlib import Path

def compute_safe_zones(zarr_base: Path, cyclone_polygons: list) -> dict:
    """Compute Green/Yellow/Red fishing zones for the Indian Ocean."""

    speed_ds = xr.open_zarr(str(zarr_base / "current_speed.zarr"))
    temp_ds = xr.open_zarr(str(zarr_base / "temperature.zarr"))

    speed = speed_ds.current_speed.isel(time=0, depth=0).values
    sst = temp_ds.temperature.isel(time=0, depth=0).values

    lats = speed_ds.latitude.values
    lons = speed_ds.longitude.values

    # Coarsen to 0.25° grid for zone computation
    step = 3  # ~0.25° at 1/12° resolution
    zones = []

    for i in range(0, len(lats), step):
        for j in range(0, len(lons), step):
            lat = float(lats[min(i, len(lats)-1)])
            lon = float(lons[min(j, len(lons)-1)])

            cell_speed = float(np.nanmean(speed[i:i+step, j:j+step]))
            cell_sst = float(np.nanmean(sst[i:i+step, j:j+step]))

            if np.isnan(cell_speed):
                continue  # Land or missing data

            # Risk scoring
            risk = 0.0
            factors = []

            if cell_speed > 1.5:
                risk += 0.5
                factors.append("high_currents")
            elif cell_speed > 0.5:
                risk += 0.2
                factors.append("moderate_currents")

            if cell_sst > 29.0:
                risk += 0.2
                factors.append("elevated_sst")

            # TODO: Add cyclone proximity check against cyclone_polygons

            category = "safe" if risk < 0.3 else ("caution" if risk < 0.7 else "danger")

            zones.append({
                "lat": lat,
                "lon": lon,
                "risk_score": round(risk, 3),
                "risk_category": category,
                "factors": factors,
            })

    return zones
```

### Step 6.3 — Export Zones as GeoJSON

```python
import json

def zones_to_geojson(zones: list) -> dict:
    """Convert zone list to GeoJSON FeatureCollection."""
    features = []
    for z in zones:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [z["lon"], z["lat"]]
            },
            "properties": {
                "risk_score": z["risk_score"],
                "risk_category": z["risk_category"],
                "factors": z["factors"],
            }
        })
    return {"type": "FeatureCollection", "features": features}

# Save to fabric
geojson = zones_to_geojson(zones)
Path("fabric/ml/safe_zones.geojson").write_text(json.dumps(geojson))
print(f"✓ {len(zones)} zones exported")
```

> **Success criteria:** `fabric/ml/surge_model.json` and `fabric/ml/safe_zones.geojson` exist. Zones are classified Green/Yellow/Red across the Indian Ocean grid.

---

## Phase 7: Inference Script — Run Rakshak on New Data (Day 7-8)

> **Goal:** Build a single script that runs ALL models on the latest data and produces the outputs the backend developer needs.
> **LeherGuide reference:** Section [20](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L1386) (Rakshak Architecture)

### Step 7.1 — Full Rakshak Inference Pipeline

```python
# services/jal-chakra/core/rakshak.py
import xgboost as xgb
import numpy as np
import xarray as xr
import json
from pathlib import Path
from datetime import datetime, timezone

def run_rakshak(zarr_base: Path):
    """Run the full Rakshak Intelligence pipeline on the latest dataset."""

    print("═══ RAKSHAK INTELLIGENCE SCAN ═══")
    timestamp = datetime.now(timezone.utc).isoformat()

    # Load models
    cyclone_model = xgb.XGBClassifier()
    cyclone_model.load_model("fabric/ml/anomaly_model.json")

    surge_model = xgb.XGBRegressor()
    surge_model.load_model("fabric/ml/surge_model.json")

    # Open latest Zarr data
    temp = xr.open_zarr(str(zarr_base / "temperature.zarr"))
    speed = xr.open_zarr(str(zarr_base / "current_speed.zarr"))
    ssh = xr.open_zarr(str(zarr_base / "sea_surface_height.zarr"))

    # Extract surface features for the entire grid
    sst = temp.temperature.isel(time=0).sel(depth=0.5, method="nearest").values
    t100 = temp.temperature.isel(time=0).sel(depth=100, method="nearest").values
    spd = speed.current_speed.isel(time=0).sel(depth=0.5, method="nearest").values
    zos_grid = ssh.sea_surface_height.isel(time=0).values.squeeze()

    lats = temp.latitude.values
    lons = temp.longitude.values

    hazards = []
    event_id = 0

    # Scan every 0.5° (every 6th cell at 1/12°) for cyclone probability
    for i in range(0, len(lats), 6):
        for j in range(0, len(lons), 6):
            s = sst[i, j]
            t1 = t100[i, j] if i < t100.shape[0] and j < t100.shape[1] else np.nan
            sp = spd[i, j]
            z = zos_grid[i, j] if i < zos_grid.shape[0] and j < zos_grid.shape[1] else np.nan

            if any(np.isnan([s, sp])):
                continue

            thermal_contrast = s - t1 if not np.isnan(t1) else 0
            vorticity = 0  # Simplified — full version uses finite differences

            features = np.array([[s, t1 if not np.isnan(t1) else s, thermal_contrast, sp, vorticity, z if not np.isnan(z) else 0]])

            try:
                prob = cyclone_model.predict_proba(features)[0][1]
            except Exception:
                continue

            if prob > 0.65:
                event_id += 1
                hazards.append({
                    "event_id": f"CYC-{event_id:04d}",
                    "type": "cyclone",
                    "latitude": float(lats[i]),
                    "longitude": float(lons[j]),
                    "probability": round(float(prob), 4),
                    "sst": round(float(s), 2),
                    "current_speed": round(float(sp), 3),
                    "detected_at": timestamp,
                })

    # Save hazards to Parquet
    import polars as pl
    if hazards:
        hazards_df = pl.DataFrame(hazards)
        hazards_df.write_parquet("fabric/ml/hazards.parquet")
        print(f"⚠ DETECTED {len(hazards)} hazard cells")
    else:
        print("✓ No hazards detected — Indian Ocean is calm")

    # Compute safe zones
    from .safe_zones import compute_safe_zones, zones_to_geojson
    zones = compute_safe_zones(zarr_base, hazards)
    geojson = zones_to_geojson(zones)
    Path("fabric/ml/safe_zones.geojson").write_text(json.dumps(geojson))
    print(f"✓ {len(zones)} fishing zones computed")

    print("═══ RAKSHAK SCAN COMPLETE ═══")

    return {
        "hazards": hazards,
        "zone_count": len(zones),
        "timestamp": timestamp,
    }
```

> **Success criteria:** Running `run_rakshak(zarr_base)` produces `hazards.parquet` and `safe_zones.geojson` from a single function call.

---

## Phase 8: Handover Package for Backend Developer (Day 8)

> **Goal:** Package everything the backend developer needs to build the FastAPI server.
> **LeherGuide reference:** Sections [10](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L604) (DuckDB), [12](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L731) (FastAPI Spec), [13](file:///c:/Users/Deep%20Saha/Desktop/Leher/LeherGuide.md#L1042) (Arrow IPC)

### What You Hand Over

```
fabric/                              ← THE DATA (your primary deliverable)
├── datasets/
│   ├── glorys/v20240101/
│   │   ├── manifest.json            ← Dataset metadata
│   │   ├── temperature.zarr/        ← Chunked Zarr arrays
│   │   ├── salinity.zarr/
│   │   ├── u_velocity.zarr/
│   │   ├── v_velocity.zarr/
│   │   ├── current_speed.zarr/      ← Derived product
│   │   └── sea_surface_height.zarr/
│   └── argo/v20240115/
│       ├── manifest.json
│       └── profiles.parquet         ← Argo float profiles
├── ml/
│   ├── anomaly_model.json           ← Trained XGBoost cyclone model
│   ├── surge_model.json             ← Trained XGBoost surge model
│   ├── hazards.parquet              ← Latest Rakshak output
│   └── safe_zones.geojson           ← Latest fishing zone classification

services/jal-chakra/                 ← THE CODE
├── core/
│   ├── pipeline.py                  ← Full Jal Chakra orchestrator
│   ├── validator.py                 ← Pariksha validation
│   ├── normalizer.py                ← Rupantar NetCDF → Zarr
│   ├── catalog.py                   ← DuckDB catalog loader
│   ├── safe_zones.py                ← Zone classification
│   └── rakshak.py                   ← Full ML inference pipeline
└── plugins/
    ├── copernicus/download.py       ← Data acquisition
    └── argo/download.py             ← Argo acquisition
```

### Instructions You Give the Backend Developer

> "Everything is in the `fabric/` directory. Here's how to use it:"

**1. To load the catalog:**
```python
from services.jal_chakra.core.catalog import init_catalog
con = init_catalog()  # Returns a DuckDB connection with all data indexed
```

**2. To read a depth slice:**
```python
import xarray as xr
ds = xr.open_zarr("fabric/datasets/glorys/v20240101/temperature.zarr")
slice_2d = ds.temperature.sel(depth=100, method="nearest").isel(time=0)
# This is the numpy array that goes into Arrow IPC encoding
```

**3. To get ML hazards:**
```python
import polars as pl
hazards = pl.read_parquet("fabric/ml/hazards.parquet")
zones = json.loads(Path("fabric/ml/safe_zones.geojson").read_text())
```

**4. To run the inference on new data:**
```python
from services.jal_chakra.core.rakshak import run_rakshak
result = run_rakshak(Path("fabric/datasets/glorys/v20240101"))
```

**5. LeherGuide reference for API spec:**
Tell them to read Section 12 (API Routes) and Section 13 (Arrow IPC encoding) of `LeherGuide.md`.

---

## Summary Checklist

| Phase | Deliverable | Success Criteria |
|---|---|---|
| 1 | First NetCDF downloaded | `fabric/raw/copernicus/glorys/raw_glorys_20240101.nc` exists (~400MB) |
| 2 | Jal Chakra pipeline runs | `fabric/datasets/glorys/v20240101/manifest.json` with `is_active: true` |
| 3 | Argo profiles ingested | `fabric/datasets/argo/v20240115/profiles.parquet` exists |
| 4 | DuckDB catalog works | SQL queries return datasets and profiles in microseconds |
| 5 | Cyclone model trained | `fabric/ml/anomaly_model.json` exists, AUC-ROC > 0.80 |
| 6 | Surge model + safe zones | `fabric/ml/surge_model.json` + `safe_zones.geojson` exist |
| 7 | Rakshak inference runs | Single `run_rakshak()` call produces all ML outputs |
| 8 | Handover package ready | Backend dev can load catalog + read Zarr + query hazards |
