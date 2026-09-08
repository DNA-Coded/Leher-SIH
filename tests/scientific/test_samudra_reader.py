import sys
import pytest
import xarray as xr
import numpy as np
import polars as pl
import json
import os
from pathlib import Path

# Fix import path for jal-chakra
jal_chakra_path = str(Path(__file__).resolve().parent.parent.parent / "services" / "jal-chakra")
if jal_chakra_path not in sys.path:
    sys.path.insert(0, jal_chakra_path)

from core.samudra_reader import (
    get_active_glorys_dataset,
    read_samudra_slice,
    get_nearest_native_depth,
    NoActiveDatasetError,
    MultipleActiveDatasetsError,
    MissingZarrVariableError,
    InvalidDepthError,
    InvalidTimeError
)
from core.safe_zones import load_surface_snapshot
from core.rakshak import build_inference_features

@pytest.fixture
def mock_fabric(tmp_path):
    """Creates a small mock Samudra Fabric with one active dataset."""
    fabric_dir = tmp_path / "fabric"
    ds_dir = fabric_dir / "datasets" / "glorys" / "v1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    # Write manifest
    manifest = {
        "dataset_id": "glorys_v1",
        "is_active": True,
        "native_depths": [0.49, 1.5, 5.0, 92.3, 109.7, 500.0],
        "variables": [
            {"canonical_name": "temperature"},
            {"canonical_name": "salinity"},
            {"canonical_name": "u_velocity"},
            {"canonical_name": "v_velocity"},
            {"canonical_name": "current_speed"}
        ]
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest))
    
    # Create tiny Zarr arrays
    lat = [10.0, 10.25]
    lon = [80.0, 80.25]
    depth = [0.49, 109.7]
    time = np.array(['2024-01-01T00:00:00', '2024-01-02T00:00:00'], dtype='datetime64[ns]')
    
    for var in manifest["variables"]:
        name = var["canonical_name"]
        data = np.random.rand(2, 2, 2, 2).astype(np.float32)
        # Add a specific NaN to test NaN preservation
        if name == "temperature":
            data[1, 0, 0, 0] = np.nan
            
        da = xr.DataArray(
            data,
            dims=["time", "depth", "latitude", "longitude"],
            coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon},
            name=name
        )
        ds = da.to_dataset()
        ds.to_zarr(ds_dir / f"{name}.zarr", consolidated=True)
        
    return fabric_dir

def test_active_dataset_resolution(mock_fabric):
    """1. Active revision resolution."""
    path, manifest = get_active_glorys_dataset(mock_fabric)
    assert path.name == "v1"
    assert manifest["is_active"] is True

def test_no_active_dataset(mock_fabric):
    """2. No active revision -> clear failure."""
    manifest_path = mock_fabric / "datasets" / "glorys" / "v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["is_active"] = False
    manifest_path.write_text(json.dumps(manifest))
    
    with pytest.raises(NoActiveDatasetError):
        get_active_glorys_dataset(mock_fabric)

def test_multiple_active_datasets(mock_fabric):
    """3. Multiple active revisions -> clear failure."""
    v2_dir = mock_fabric / "datasets" / "glorys" / "v2"
    v2_dir.mkdir(parents=True)
    manifest_path = mock_fabric / "datasets" / "glorys" / "v1" / "manifest.json"
    (v2_dir / "manifest.json").write_text(manifest_path.read_text())
    
    with pytest.raises(MultipleActiveDatasetsError):
        get_active_glorys_dataset(mock_fabric)

def test_depth_resolution():
    """4. Requested 100m resolves to nearest native depth."""
    native_depths = [0.49, 1.5, 92.3, 109.7, 500.0]
    # Nearest to 100 is 109.7 (diff is 9.7 vs 7.7) wait, 100 - 92.3 = 7.7. 109.7 - 100 = 9.7.
    # Nearest is 92.3!
    assert get_nearest_native_depth(100.0, native_depths) == 92.3
    
def test_negative_depth_fails():
    """5. Depth remains positive-down."""
    with pytest.raises(InvalidDepthError):
        get_nearest_native_depth(-10.0, [0.49, 10.0])

def test_reader_slice(mock_fabric):
    """6. Surface layer selection works. 7. 100m layer selection works. 8. Time selection uses dataset time. 11. NaN values remain NaN."""
    dataset_path, manifest = get_active_glorys_dataset(mock_fabric)
    
    # Test surface
    ds_surface = read_samudra_slice(dataset_path, manifest, ["temperature"], depth_m=0.0, time_str="latest")
    assert ds_surface.depth.values == 0.49
    # Time should be latest (2024-01-02)
    assert str(ds_surface.time.values) == "2024-01-02T00:00:00.000000000"
    # NaN is preserved at [1, 0, 0, 0]
    assert np.isnan(ds_surface.temperature.values[0, 0])
    
    # Test 100m
    ds_100m = read_samudra_slice(dataset_path, manifest, ["temperature"], depth_m=100.0, time_str="2024-01-01T00:00:00.000000000")
    assert ds_100m.depth.values == 109.7
    assert str(ds_100m.time.values) == "2024-01-01T00:00:00.000000000"

def test_missing_variable_explicit(mock_fabric):
    """12. Missing optional variable behavior is explicit."""
    dataset_path, manifest = get_active_glorys_dataset(mock_fabric)
    with pytest.raises(MissingZarrVariableError):
        read_samudra_slice(dataset_path, manifest, ["fake_variable"])

def test_rakshak_input_compatibility(mock_fabric):
    """15. Test Rakshak input compatibility."""
    # load_surface_snapshot now uses our mock fabric
    surface_df = load_surface_snapshot(mock_fabric)
    
    # Must have latitude, longitude, thetao, uo, vo, surface_salinity, temp_100m, thermal_contrast
    expected_cols = {"latitude", "longitude", "thetao", "uo", "vo", "surface_salinity", "temp_100m", "thermal_contrast"}
    assert expected_cols.issubset(set(surface_df.columns))
    
    # We coarsen it
    from core.safe_zones import coarsen_to_grid
    grid_df = coarsen_to_grid(surface_df)
    
    # Must have lat_grid, lon_grid, sst, uo, vo, current_speed, temp_100m, thermal_contrast
    expected_grid = {"lat_grid", "lon_grid", "sst", "uo", "vo", "current_speed", "temp_100m", "thermal_contrast", "surface_salinity"}
    assert expected_grid.issubset(set(grid_df.columns))
    
    # Test build_inference_features
    features_df = build_inference_features(grid_df)
    
    # Must have sst, temp_100m, thermal_contrast, current_speed
    assert "sst" in features_df.columns
    assert "temp_100m" in features_df.columns
    assert "thermal_contrast" in features_df.columns
    assert "current_speed" in features_df.columns
    assert "vorticity" in features_df.columns
    assert "zos" in features_df.columns
    
    # NaN/null is not coerced!
    # Because we injected a NaN at [1,0,0,0], thetao should be NaN/null for that cell
    # It should propagate
    nan_count = features_df.select((pl.col("sst").is_null() | pl.col("sst").is_nan()).sum()).item()
    assert nan_count > 0
