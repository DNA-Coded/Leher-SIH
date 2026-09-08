"""
validator.py — Pariksha (Examination) Stage
Leher / Jal Chakra Pipeline

Validates a raw GLORYS NetCDF file before transformation.
All failures are logged to fabric/quarantined/errors.jsonl.
NaN must NEVER become 0.0 — this check is non-negotiable.
"""
import json
import xarray as xr
from pathlib import Path


class ValidationError(Exception):
    """Raised when a NetCDF fails Pariksha validation."""


def validate_glorys(nc_path: Path) -> xr.Dataset:
    """
    Pariksha: Validate a raw GLORYS NetCDF file.

    Checks:
    1. Required variables present
    2. Coordinate ranges are physically sane
    3. Depth uses positive-down convention
    4. Temperature units are degrees_C
    5. No variable has _FillValue = 0.0  (scientifically catastrophic)

    Returns the opened xr.Dataset on success.
    Raises ValidationError and writes to quarantine on failure.
    """
    ds = xr.open_dataset(nc_path)
    errors = []

    # 1. Required variables — zos is optional (not always present in training data)
    core_required = {"thetao", "so", "uo", "vo"}
    missing = core_required - set(ds.data_vars)
    if missing:
        errors.append(f"Missing required variables: {sorted(missing)}")

    # 2. Coordinate sanity
    if "latitude" in ds.coords:
        lat_min = float(ds.latitude.min())
        lat_max = float(ds.latitude.max())
        if lat_min < -90 or lat_max > 90:
            errors.append(f"Latitude out of range: [{lat_min}, {lat_max}]")
    else:
        errors.append("No 'latitude' coordinate found")

    if "longitude" in ds.coords:
        lon_min = float(ds.longitude.min())
        lon_max = float(ds.longitude.max())
        if lon_min < -180 or lon_max > 360:
            errors.append(f"Longitude out of range: [{lon_min}, {lon_max}]")
    else:
        errors.append("No 'longitude' coordinate found")

    # 3. Depth must be positive-down
    if "depth" in ds.coords:
        if float(ds.depth.min()) < 0:
            errors.append(
                "Depth values are negative — expected positive-down convention. "
                "Check if the file uses pressure levels instead."
            )
    else:
        errors.append("No 'depth' coordinate found")

    # 4. Temperature units
    if "thetao" in ds.data_vars:
        units = getattr(ds.thetao, "units", None)
        if units is not None and units not in ("degrees_C", "degC", "degree_Celsius", "°C"):
            errors.append(f"Temperature units unexpected: '{units}' — expected degrees_C")

    # 5. Fill-value safety — 0.0 fill would silently corrupt all ocean values
    for var in ds.data_vars:
        fv = ds[var].encoding.get("_FillValue", None)
        if fv is not None and float(fv) == 0.0:
            errors.append(
                f"Variable '{var}' has _FillValue=0.0 — this is scientifically "
                "non-negotiable. NaN values would become zero in the output."
            )

    if errors:
        _quarantine(nc_path, errors)
        raise ValidationError(
            f"Pariksha FAILED for {nc_path.name}:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    print(f"[PARIKSHA] PASS: {nc_path.name}")
    return ds


def _quarantine(nc_path: Path, errors: list):
    """Write a quarantine error record so failures are traceable."""
    record = {"file": str(nc_path), "errors": errors}
    log = Path("fabric/quarantined/errors.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[PARIKSHA] Quarantined: {nc_path.name} -> {log}")
