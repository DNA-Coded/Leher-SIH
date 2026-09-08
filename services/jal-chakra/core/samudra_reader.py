import json
from pathlib import Path
import xarray as xr
import polars as pl
import numpy as np
from datetime import datetime, timezone

class NoActiveDatasetError(Exception):
    pass

class MultipleActiveDatasetsError(Exception):
    pass

class MissingZarrVariableError(Exception):
    pass

class InvalidDepthError(Exception):
    pass

class InvalidTimeError(Exception):
    pass

def get_active_glorys_dataset(fabric_dir: Path = Path("fabric")) -> tuple[Path, dict]:
    """
    Scans the GLORYS dataset revisions to find the single active dataset.
    Returns (dataset_path, manifest_dict).
    """
    glorys_dir = fabric_dir / "datasets" / "glorys"
    if not glorys_dir.exists():
        raise NoActiveDatasetError(f"No glorys datasets directory found at {glorys_dir}")

    active_paths = []
    active_manifest = None

    for rev_dir in glorys_dir.iterdir():
        if not rev_dir.is_dir():
            continue
        manifest_path = rev_dir / "manifest.json"
        if not manifest_path.exists():
            continue
            
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if manifest.get("is_active"):
                active_paths.append(rev_dir)
                active_manifest = manifest
        except Exception:
            pass
            
    if len(active_paths) == 0:
        raise NoActiveDatasetError("No active GLORYS dataset found in manifests.")
    elif len(active_paths) > 1:
        raise MultipleActiveDatasetsError(f"Found {len(active_paths)} active datasets! Ambiguous state.")
        
    return active_paths[0], active_manifest

def get_nearest_native_depth(requested_depth: float, native_depths: list[float]) -> float:
    """Finds the nearest available depth from the manifest's native depths."""
    if not native_depths:
        raise InvalidDepthError("Manifest has no native depths available.")
    
    # Must be positive-down
    if requested_depth < 0:
        raise InvalidDepthError("Requested depth must be positive-down.")
        
    idx = np.argmin(np.abs(np.array(native_depths) - requested_depth))
    return native_depths[idx]

def read_samudra_slice(
    dataset_path: Path,
    manifest: dict,
    variables: list[str],
    depth_m: float = None,
    time_str: str = None,
    min_lat: float = None,
    max_lat: float = None,
    min_lon: float = None,
    max_lon: float = None
) -> xr.Dataset:
    """
    Reads a slice from the active Zarr dataset.
    Returns an xarray Dataset with requested variables.
    """
    datasets = []
    
    for var in variables:
        zarr_path = dataset_path / f"{var}.zarr"
        if not zarr_path.exists():
            raise MissingZarrVariableError(f"Variable {var} not found at {zarr_path}")
            
        ds = xr.open_zarr(zarr_path, consolidated=True)
        
        # Slicing time
        if time_str is not None:
            if time_str == "latest":
                ds = ds.isel(time=-1)
            else:
                times_str_array = ds.time.values.astype(str)
                # sometimes xarray time values are datetime64 strings like '2024-01-01T00:00:00.000000000'
                if time_str not in times_str_array:
                    raise InvalidTimeError(f"Time {time_str} not in dataset time coordinates.")
                time_idx = np.where(times_str_array == time_str)[0][0]
                ds = ds.isel(time=time_idx)
        else:
            ds = ds.isel(time=-1) # Default to latest
            
        # Slicing depth
        if depth_m is not None and "depth" in ds.coords:
            native_depths = manifest.get("native_depths", [])
            nearest = get_nearest_native_depth(depth_m, native_depths)
            
            # Find the index of the nearest depth
            depth_vals = ds.depth.values
            idx = np.argmin(np.abs(depth_vals - nearest))
            ds = ds.isel(depth=idx)
            
        # Slicing spatial bounds
        if min_lat is not None and max_lat is not None and "latitude" in ds.coords:
            ds = ds.sel(latitude=slice(min_lat, max_lat))
            
        if min_lon is not None and max_lon is not None and "longitude" in ds.coords:
            ds = ds.sel(longitude=slice(min_lon, max_lon))
            
        datasets.append(ds)
        
    return xr.merge(datasets)
