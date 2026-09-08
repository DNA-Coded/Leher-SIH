import os
from pathlib import Path

# Repo root is 4 levels up: apps/api/leher/paths.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

FABRIC_ML_DIR = REPO_ROOT / "fabric" / "ml"
HAZARDS_PARQUET = FABRIC_ML_DIR / "hazards.parquet"
SAFE_ZONES_GEOJSON = FABRIC_ML_DIR / "safe_zones.geojson"
ECOSYSTEM_JSON = FABRIC_ML_DIR / "ecosystem.json"
ADVISORIES_JSON = FABRIC_ML_DIR / "advisories.json"
RAKSHAK_RUN_JSON = FABRIC_ML_DIR / "rakshak_run.json"
