import pytest
from pathlib import Path
import xarray as xr
import numpy as np
import json
import sys

# Ensure jal_chakra is importable
REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
jal_chakra_path = REPO_ROOT / "services" / "jal-chakra"
if str(jal_chakra_path) not in sys.path:
    sys.path.insert(0, str(jal_chakra_path))

from plugins.copernicus.client import acquire_glorys_data
from core.pipeline import run_jal_chakra
from core.validator import ValidationError

def create_synthetic_netcdf(output_path: Path, bad_fill_value: bool = False, missing_var: bool = False, bad_depth: bool = False):
    """Creates a mock GLORYS netcdf file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Dimensions
    lat = np.array([10.0, 10.5])
    lon = np.array([80.0, 80.5])
    depth = np.array([-5.0, -10.0]) if bad_depth else np.array([0.5, 5.0])
    time = np.array(['2024-01-01T00:00:00'], dtype='datetime64[ns]')
    
    coords = {"time": time, "depth": depth, "latitude": lat, "longitude": lon}
    
    ds = xr.Dataset(coords=coords)
    
    # Vars
    vars_to_add = ["thetao", "so", "uo", "vo", "zos", "mlotst", "bottomT", "siconc", "sithick", "chl"]
    if missing_var:
        vars_to_add.remove("thetao")
        
    for var in vars_to_add:
        data = np.ones((1, 2, 2, 2), dtype=np.float32)
        da = xr.DataArray(data, dims=["time", "depth", "latitude", "longitude"], coords=coords)
        if var == "thetao":
            da.attrs["units"] = "degrees_C"
            
        ds[var] = da
        if bad_fill_value:
            ds[var].encoding["_FillValue"] = 0.0
        else:
            ds[var].encoding["_FillValue"] = -32767.0
            
    # Need CF global attr
    ds.attrs["Conventions"] = "CF-1.8"
    
    ds.to_netcdf(output_path)
    return output_path

def test_full_pipeline_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fabric_dir = tmp_path / "fabric"
    
    # Mock copernicusmarine.subset to just create our synthetic netcdf
    def mock_subset(*args, **kwargs):
        out_dir = Path(kwargs.get("output_directory"))
        out_file = kwargs.get("output_filename")
        create_synthetic_netcdf(out_dir / out_file)
        
    import copernicusmarine
    monkeypatch.setattr(copernicusmarine, "subset", mock_subset)
    
    # 1. UTPATTI
    raw_dir = fabric_dir / "raw" / "copernicus" / "glorys"
    nc_path = acquire_glorys_data(
        output_dir=raw_dir,
        start_datetime="2024-01-01T00:00:00",
        end_datetime="2024-01-01T23:59:59",
    )
    
    assert nc_path.exists()
    
    # 2-4. PARIKSHA, RUPANTAR, PRAKASHAN
    manifest = run_jal_chakra(nc_path, "glorys", "v20240101")
    
    assert manifest["dataset_id"] == "glorys_v20240101"
    assert manifest["is_active"] is True
    
    # Check Zarrs were written
    ds_dir = fabric_dir / "datasets" / "glorys" / "v20240101"
    assert (ds_dir / "manifest.json").exists()
    assert (ds_dir / "temperature.zarr").exists()
    assert (ds_dir / "current_speed.zarr").exists()
    assert (ds_dir / "bottom_temperature.zarr").exists()
    assert (ds_dir / "sea_ice_fraction.zarr").exists()
    assert (ds_dir / "sea_ice_thickness.zarr").exists()
    assert (ds_dir / "sea_surface_height.zarr").exists()
    assert (ds_dir / "mixed_layer_depth.zarr").exists()
    assert (ds_dir / "salinity.zarr").exists()

def test_pipeline_validation_failure_bad_depth(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fabric_dir = tmp_path / "fabric"
    
    def mock_subset(*args, **kwargs):
        create_synthetic_netcdf(Path(kwargs.get("output_directory")) / kwargs.get("output_filename"), bad_depth=True)
        
    import copernicusmarine
    monkeypatch.setattr(copernicusmarine, "subset", mock_subset)
    
    raw_dir = fabric_dir / "raw" / "copernicus" / "glorys"
    nc_path = acquire_glorys_data(raw_dir, "2024-01-01T00:00:00", "2024-01-01T23:59:59")
    
    with pytest.raises(ValidationError, match="Depth values are negative"):
        run_jal_chakra(nc_path, "glorys", "v20240101")
        
    # Check quarantined
    assert (fabric_dir / "quarantined" / "errors.jsonl").exists()

def test_pipeline_validation_failure_bad_fill_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fabric_dir = tmp_path / "fabric"
    
    def mock_subset(*args, **kwargs):
        create_synthetic_netcdf(Path(kwargs.get("output_directory")) / kwargs.get("output_filename"), bad_fill_value=True)
        
    import copernicusmarine
    monkeypatch.setattr(copernicusmarine, "subset", mock_subset)
    
    raw_dir = fabric_dir / "raw" / "copernicus" / "glorys"
    nc_path = acquire_glorys_data(raw_dir, "2024-01-01T00:00:00", "2024-01-01T23:59:59")
    
    with pytest.raises(ValidationError, match="scientifically non-negotiable"):
        run_jal_chakra(nc_path, "glorys", "v20240101")

def test_pipeline_atomic_activation(tmp_path, monkeypatch):
    # Test that previous revision gets deactivated
    monkeypatch.chdir(tmp_path)
    fabric_dir = tmp_path / "fabric"
    
    def mock_subset(*args, **kwargs):
        create_synthetic_netcdf(Path(kwargs.get("output_directory")) / kwargs.get("output_filename"))
        
    import copernicusmarine
    monkeypatch.setattr(copernicusmarine, "subset", mock_subset)
    
    raw_dir = fabric_dir / "raw" / "copernicus" / "glorys"
    
    # 1st run
    nc1 = acquire_glorys_data(raw_dir, "2024-01-01T00:00:00", "2024-01-01T23:59:59", "file1.nc")
    run_jal_chakra(nc1, "glorys", "v1")
    
    # 2nd run
    nc2 = acquire_glorys_data(raw_dir, "2024-01-02T00:00:00", "2024-01-02T23:59:59", "file2.nc")
    run_jal_chakra(nc2, "glorys", "v2")
    
    # Check states
    ds1_manifest = json.loads((fabric_dir / "datasets" / "glorys" / "v1" / "manifest.json").read_text())
    ds2_manifest = json.loads((fabric_dir / "datasets" / "glorys" / "v2" / "manifest.json").read_text())
    
    assert ds1_manifest["is_active"] is False
    assert ds2_manifest["is_active"] is True
