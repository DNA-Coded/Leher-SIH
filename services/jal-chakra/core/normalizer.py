"""
normalizer.py — Rupantar (Transformation) Stage
Leher / Jal Chakra Pipeline

Converts a validated GLORYS xr.Dataset into:
- Chunked Zarr arrays in the Samudra Data Fabric (for visualization)
- A flat Parquet file per variable group (for ML training)

NaN MUST remain NaN — never coerced to 0.0.
Chunk shape: (1 time, 1 depth, full_lat, full_lon).
"""
import json
import hashlib
import numpy as np
import xarray as xr
from pathlib import Path


def convert_to_zarr(ds: xr.Dataset, source_name: str, revision: str) -> tuple[Path, list]:
    """
    Rupantar: Convert a validated xr.Dataset to chunked Zarr.

    Each variable gets its own .zarr store.
    Also computes derived product: current_speed = sqrt(uo² + vo²).

    Returns (output_base_path, manifest_vars_list).
    """
    output_base = Path(f"fabric/datasets/{source_name}/{revision}")
    output_base.mkdir(parents=True, exist_ok=True)

    # Canonical variable names (Leher-internal, not GLORYS names)
    var_map = {
        "thetao": "temperature",
        "so": "salinity",
        "uo": "u_velocity",
        "vo": "v_velocity",
        "zos": "sea_surface_height",
        "mlotst": "mixed_layer_depth",
        "bottomT": "bottom_temperature",
        "siconc": "sea_ice_fraction",
        "sithick": "sea_ice_thickness",
        "chl": "chlorophyll",
    }

    manifest_vars = []

    for src_name, canon_name in var_map.items():
        if src_name not in ds.data_vars:
            continue

        da = ds[src_name]

        # float32 + NaN fill — the golden rule
        data = da.values.astype(np.float32)
        # Replace fill values / inf with NaN, never with 0
        data = np.where(np.isfinite(data), data, np.float32("nan"))

        clean_da = xr.DataArray(
            data,
            dims=da.dims,
            coords=da.coords,
            name=canon_name,
            attrs=da.attrs,
        )

        zarr_path = output_base / f"{canon_name}.zarr"

        # Chunk: 1 per time step, 1 per depth level, full spatial extent
        chunks = {}
        for dim in clean_da.dims:
            if dim == "time":
                chunks[dim] = 1
            elif dim == "depth":
                chunks[dim] = 1
            else:
                chunks[dim] = clean_da.sizes[dim]  # full lat / full lon

        chunk_tuple = tuple(chunks[d] for d in clean_da.dims)

        clean_da.to_dataset().to_zarr(
            str(zarr_path),
            mode="w",
            consolidated=True,
            encoding={
                canon_name: {
                    "chunks": chunk_tuple,
                    "dtype": "float32",
                }
            },
        )

        data_finite = data[np.isfinite(data)]
        manifest_vars.append({
            "canonical_name": canon_name,
            "source_name": src_name,
            "units": str(da.attrs.get("units", "unknown")),
            "standard_name": str(da.attrs.get("standard_name", "")),
            "data_min": float(np.nanmin(data_finite)) if data_finite.size > 0 else None,
            "data_max": float(np.nanmax(data_finite)) if data_finite.size > 0 else None,
            "zarr_path": str(zarr_path.relative_to("fabric")),
        })

        print(f"  [RUPANTAR] {src_name} -> {canon_name}.zarr")

    # Derived product: current speed
    if "uo" in ds.data_vars and "vo" in ds.data_vars:
        uo = ds["uo"].values.astype(np.float32)
        vo = ds["vo"].values.astype(np.float32)
        speed = np.sqrt(uo ** 2 + vo ** 2)
        # Propagate NaN from either component
        speed = np.where(np.isfinite(uo) & np.isfinite(vo), speed, np.float32("nan"))

        speed_da = xr.DataArray(
            speed,
            dims=ds["uo"].dims,
            coords=ds["uo"].coords,
            name="current_speed",
            attrs={"units": "m s-1", "standard_name": "sea_water_speed"},
        )
        speed_path = output_base / "current_speed.zarr"
        speed_da.to_dataset().to_zarr(str(speed_path), mode="w", consolidated=True)

        speed_finite = speed[np.isfinite(speed)]
        manifest_vars.append({
            "canonical_name": "current_speed",
            "source_name": "derived(uo, vo)",
            "units": "m s-1",
            "standard_name": "sea_water_speed",
            "data_min": float(np.nanmin(speed_finite)) if speed_finite.size > 0 else None,
            "data_max": float(np.nanmax(speed_finite)) if speed_finite.size > 0 else None,
            "zarr_path": str(speed_path.relative_to("fabric")),
        })
        print(f"  [RUPANTAR] derived -> current_speed.zarr")

    return output_base, manifest_vars


def compute_checksum(directory: Path) -> str:
    """SHA-256 hash over all files in a directory (for provenance)."""
    h = hashlib.sha256()
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            h.update(f.read_bytes())
    return h.hexdigest()
