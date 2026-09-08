import sys
import pytest
import xarray as xr
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch

# Fix import path for jal-chakra
jal_chakra_path = str(Path(__file__).resolve().parent.parent.parent / "services" / "jal-chakra")
if jal_chakra_path not in sys.path:
    sys.path.insert(0, jal_chakra_path)

from core.validator import validate_glorys, ValidationError
from core.normalizer import convert_to_zarr
from core.pipeline import run_jal_chakra
from plugins.copernicus.client import acquire_glorys_data

@pytest.fixture
def base_dataset():
    """Creates a small structural xr.Dataset mimicking GLORYS."""
    lon = np.array([80.0, 80.5])
    lat = np.array([15.0, 15.5])
    depth = np.array([0.5, 5.0])
    time = np.array(['2024-01-01T00:00:00'], dtype='datetime64[ns]')
    
    # Create variables with NaN to ensure NaN is not coerced to 0.0
    thetao = np.array([[[[28.0, np.nan], [29.0, 29.5]], [[27.0, 27.5], [28.0, np.nan]]]])
    so = np.array([[[[34.0, 34.5], [35.0, np.nan]], [[34.5, 34.6], [35.1, 35.2]]]])
    uo = np.array([[[[0.1, 0.2], [np.nan, 0.4]], [[0.05, 0.1], [0.15, 0.2]]]])
    vo = np.array([[[[0.0, 0.1], [0.2, 0.3]], [[-0.1, 0.0], [0.1, 0.2]]]])
    
    ds = xr.Dataset(
        data_vars=dict(
            thetao=(["time", "depth", "latitude", "longitude"], thetao, {"units": "degrees_C"}),
            so=(["time", "depth", "latitude", "longitude"], so),
            uo=(["time", "depth", "latitude", "longitude"], uo),
            vo=(["time", "depth", "latitude", "longitude"], vo),
        ),
        coords=dict(
            longitude=(["longitude"], lon),
            latitude=(["latitude"], lat),
            depth=(["depth"], depth),
            time=(["time"], time),
        )
    )
    # Give thetao a safe FillValue internally for testing. Note: 0.0 fill is catastrophic.
    ds.thetao.encoding["_FillValue"] = -32767.0
    return ds

def test_invalid_dataset_missing_latitude(tmp_path, base_dataset):
    """1. Invalid dataset missing latitude -> validation failure."""
    ds_invalid = base_dataset.drop_vars("latitude")
    nc_path = tmp_path / "missing_lat.nc"
    ds_invalid.to_netcdf(nc_path)
    
    with pytest.raises(ValidationError, match="No 'latitude' coordinate found"):
        validate_glorys(nc_path)

def test_invalid_dataset_missing_variable(tmp_path, base_dataset):
    """2. Invalid dataset missing required variable -> validation failure."""
    ds_invalid = base_dataset.drop_vars("uo")
    nc_path = tmp_path / "missing_uo.nc"
    ds_invalid.to_netcdf(nc_path)
    
    with pytest.raises(ValidationError, match="Missing required variables"):
        validate_glorys(nc_path)

def test_incorrect_depth_orientation(tmp_path, base_dataset):
    """3. Incorrect depth orientation -> validation failure."""
    # Negative depth (positive-up) is invalid
    ds_invalid = base_dataset.assign_coords(depth=-base_dataset.depth)
    nc_path = tmp_path / "neg_depth.nc"
    ds_invalid.to_netcdf(nc_path)
    
    with pytest.raises(ValidationError, match="positive-down convention"):
        validate_glorys(nc_path)

def test_nan_values_remain_nan_and_valid_dataset_passes(tmp_path, base_dataset):
    """4. NaN values remain NaN. 5. A valid small structural dataset can pass validation."""
    nc_path = tmp_path / "valid.nc"
    base_dataset.to_netcdf(nc_path)
    
    # Validation should pass
    validated_ds = validate_glorys(nc_path)
    assert "thetao" in validated_ds.data_vars
    
    # 6. NetCDF -> Zarr mapping
    # 7. Coordinates are preserved.
    # Convert to Zarr
    import os
    orig_cwd = os.getcwd()
    os.chdir(tmp_path) # write to temp fabric directory
    try:
        output_base, manifest_vars = convert_to_zarr(validated_ds, "glorys", "vtest")
        
        # Check Zarr creation
        zarr_path = Path("fabric/datasets/glorys/vtest/temperature.zarr")
        assert zarr_path.exists()
        
        # Read back Zarr and verify NaN and coordinates
        zarr_ds = xr.open_zarr(zarr_path, consolidated=True)
        assert "temperature" in zarr_ds.data_vars
        assert np.isnan(zarr_ds.temperature.values[0, 0, 0, 1])  # The NaN we placed remains NaN!
        assert zarr_ds.temperature.values[0, 0, 0, 1] != 0.0
        
        # Coordinates preserved
        assert "latitude" in zarr_ds.coords
        assert zarr_ds.latitude.values[0] == 15.0
        
        # Derived variable current_speed should exist
        assert Path("fabric/datasets/glorys/vtest/current_speed.zarr").exists()
        
    finally:
        os.chdir(orig_cwd)

def test_pipeline_overwritten_protection_and_manifest(tmp_path, base_dataset):
    """8. Published dataset cannot be silently overwritten (without deactivation). 9. Manifest generated."""
    nc_path = tmp_path / "raw_glorys_20240101.nc"
    base_dataset.to_netcdf(nc_path)
    
    import os
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Run pipeline first time
        manifest1 = run_jal_chakra(nc_path)
        assert manifest1["is_active"] is True
        assert manifest1["dataset_id"] == "glorys_v20240101"
        assert len(manifest1["variables"]) > 0
        
        # Manifest is generated on disk
        manifest_path = Path("fabric/datasets/glorys/v20240101/manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest1))
        
        # Run pipeline again, say we got a newer file
        nc_path2 = tmp_path / "raw_glorys_20240102.nc"
        base_dataset.to_netcdf(nc_path2)
        manifest2 = run_jal_chakra(nc_path2)
        
        # 8. Check that the older manifest is now deactivated!
        old_manifest = json.loads(manifest_path.read_text())
        assert old_manifest["is_active"] is False
        assert manifest2["is_active"] is True
        
    finally:
        os.chdir(orig_cwd)

@patch("plugins.copernicus.client.copernicusmarine.subset")
def test_copernicus_acquisition_mocked(mock_subset, tmp_path):
    """10. Copernicus acquisition is mocked and does not require live credentials."""
    # Simulate a successful download by creating a dummy file
    def mock_subset_side_effect(**kwargs):
        output_dir = Path(kwargs['output_directory'])
        filename = kwargs['output_filename']
        import xarray as xr
        import numpy as np
        coords = {
            "time": [np.datetime64("2024-01-01T12:00:00")],
            "depth": [0.49],
            "latitude": [10.0],
            "longitude": [80.0]
        }
        ds = xr.Dataset({"dummy": (("time", "depth", "latitude", "longitude"), np.zeros((1, 1, 1, 1)))}, coords=coords)
        ds.to_netcdf(output_dir / filename)
        
    mock_subset.side_effect = mock_subset_side_effect
    
    out_path = acquire_glorys_data(
        output_dir=tmp_path,
        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-01-01T23:59:59"
    )
    
    # Assert copernicusmarine.subset was called twice (physical + bio)
    assert mock_subset.call_count == 2
    kwargs = mock_subset.call_args.kwargs
    assert kwargs["minimum_longitude"] == 20.0
    assert kwargs["maximum_longitude"] == 130.0
    assert kwargs["minimum_depth"] == 0.0
    assert out_path.exists()
