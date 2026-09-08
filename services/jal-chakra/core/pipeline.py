"""
pipeline.py — Jal Chakra Pipeline Orchestrator
Leher / Jal Chakra Pipeline

The 4-stage pipeline:
  UTPATTI   → File exists on disk (acquired)
  PARIKSHA  → Validate the NetCDF
  RUPANTAR  → Convert to Zarr + derived products
  PRAKASHAN → Write immutable manifest, deactivate previous version

Run with:
  from services.jal_chakra.core.pipeline import run_jal_chakra
  manifest = run_jal_chakra(Path("fabric/raw/copernicus/glorys/raw_glorys_20240101.nc"))
"""
import json
from pathlib import Path
from datetime import datetime, timezone

from .validator import validate_glorys
from .normalizer import convert_to_zarr, compute_checksum


def run_jal_chakra(
    nc_path: Path,
    source_name: str = "glorys",
    revision: str = None,
) -> dict:
    """
    Run the full 4-stage Jal Chakra pipeline on a single NetCDF file.

    Args:
        nc_path:     Path to the raw .nc file (from Copernicus or similar).
        source_name: Short name for the data source (default: 'glorys').
        revision:    Version label. Auto-derived from filename if None.

    Returns:
        The manifest dict that was written to disk.
    """
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF not found: {nc_path}")

    if revision is None:
        # e.g. "raw_glorys_20240101.nc" -> "v20240101"
        stem_parts = nc_path.stem.split("_")
        revision = f"v{stem_parts[-1]}" if stem_parts else "v001"

    print("=" * 50)
    print("JAL CHAKRA PIPELINE")
    print(f"  Source : {nc_path}")
    print(f"  Revision: {revision}")
    print("=" * 50)

    # Stage 1: Utpatti (Origin) — file already on disk
    print("\n[1/4] UTPATTI (Origin): File acquired")

    # Stage 2: Pariksha (Examination)
    print("\n[2/4] PARIKSHA (Examination): Validating...")
    ds = validate_glorys(nc_path)

    # Stage 3: Rupantar (Transformation)
    print("\n[3/4] RUPANTAR (Transformation): Converting to Zarr...")
    output_base, manifest_vars = convert_to_zarr(ds, source_name, revision)

    # Stage 4: Prakashan (Illumination)
    print("\n[4/4] PRAKASHAN (Illumination): Publishing manifest...")

    # Deactivate any previous active version for this source
    datasets_dir = Path("fabric/datasets") / source_name
    for old_manifest_path in datasets_dir.glob("*/manifest.json"):
        try:
            old = json.loads(old_manifest_path.read_text())
            if old.get("is_active"):
                old["is_active"] = False
                old_manifest_path.write_text(json.dumps(old, indent=2))
                print(f"  Deactivated previous: {old.get('revision_label', '?')}")
        except Exception:
            pass  # Don't crash if an old manifest is malformed

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
        },
    }

    manifest_path = output_base / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest written: {manifest_path}")

    print()
    print("=" * 50)
    print("PIPELINE COMPLETE")
    print(f"  Active dataset: {manifest['dataset_id']}")
    print(f"  Variables: {[v['canonical_name'] for v in manifest_vars]}")
    depths = manifest.get("native_depths", [])
    print(f"  Depths: {len(depths)} levels")
    print("=" * 50)

    return manifest
