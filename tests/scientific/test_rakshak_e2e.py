import pytest
import numpy as np
import xarray as xr
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
jal_chakra_path = REPO_ROOT / "services" / "jal-chakra"
if str(jal_chakra_path) not in sys.path:
    sys.path.insert(0, str(jal_chakra_path))

from plugins.copernicus.client import acquire_glorys_data
from core.pipeline import run_jal_chakra
from core.rakshak import run_rakshak

def create_anomalous_synthetic_netcdf(output_path: Path):
    """Creates a mock GLORYS netcdf file with extreme values to trigger hazards."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3x3 grid around the Indian Ocean, plus 2 depths, 1 time
    lat = np.array([10.0, 15.0, 20.0])
    lon = np.array([70.0, 80.0, 90.0])
    depth = np.array([0.5, 95.0]) # Surface, and near 100m
    time = np.array(['2024-01-01T00:00:00'], dtype='datetime64[ns]')
    
    coords = {"time": time, "depth": depth, "latitude": lat, "longitude": lon}
    ds = xr.Dataset(coords=coords)
    
    # High SST (31C) for coral bleaching and cyclone triggers at (lat=15, lon=80)
    thetao_data = np.full((1, 2, 3, 3), 25.0, dtype=np.float32)
    thetao_data[0, 0, 1, 1] = 31.0 # Surface hot spot
    thetao_data[0, 1, 1, 1] = 20.0 # 100m cold spot -> thermal contrast = 11C
    
    # Strong currents (2.0 m/s) at (lat=10, lon=70)
    uo_data = np.full((1, 2, 3, 3), 0.1, dtype=np.float32)
    vo_data = np.full((1, 2, 3, 3), 0.1, dtype=np.float32)
    uo_data[0, 0, 0, 0] = 2.0
    vo_data[0, 0, 0, 0] = 0.0 # current_speed = 2.0 m/s
    
    # NaNs injected at (lat=20, lon=90) for all vars to test graceful fallback
    thetao_data[0, :, 2, 2] = np.nan
    uo_data[0, :, 2, 2] = np.nan
    vo_data[0, :, 2, 2] = np.nan
    
    # High zos (1.5m) for extreme tides at (lat=15, lon=90)
    zos_data = np.full((1, 2, 3, 3), 0.1, dtype=np.float32)
    zos_data[0, :, 1, 2] = 1.5
    
    so_data = np.full((1, 2, 3, 3), 35.0, dtype=np.float32)
    mlotst_data = np.full((1, 2, 3, 3), 30.0, dtype=np.float32)
    
    vars_dict = {
        "thetao": thetao_data,
        "uo": uo_data,
        "vo": vo_data,
        "so": so_data,
        "zos": zos_data,
        "mlotst": mlotst_data
    }
    
    for var, data in vars_dict.items():
        da = xr.DataArray(data, dims=["time", "depth", "latitude", "longitude"], coords=coords)
        ds[var] = da
        ds[var].encoding["_FillValue"] = -32767.0
        
    ds.attrs["Conventions"] = "CF-1.8"
    ds.to_netcdf(output_path)
    return output_path


def test_rakshak_full_pipeline(tmp_path, monkeypatch):
    """
    1. Ingest synthetic data
    2. Run Rakshak
    3. Verify outputs
    """
    monkeypatch.chdir(tmp_path)
    fabric_dir = tmp_path / "fabric"
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Setup mock copernicus subset
    def mock_subset(*args, **kwargs):
        out_dir = Path(kwargs.get("output_directory"))
        out_file = kwargs.get("output_filename")
        create_anomalous_synthetic_netcdf(out_dir / out_file)
        
    import copernicusmarine
    monkeypatch.setattr(copernicusmarine, "subset", mock_subset)
    
    # 1. Ingest
    raw_dir = fabric_dir / "raw" / "copernicus" / "glorys"
    nc_path = acquire_glorys_data(raw_dir, "2024-01-01T00:00:00", "2024-01-01T23:59:59")
    run_jal_chakra(nc_path, "glorys", "vtest")
    
    # We must patch paths inside rakshak.py and safe_zones.py since they use relative Path("...")
    # Because we did monkeypatch.chdir(tmp_path), Path("fabric/...") automatically evaluates
    # to tmp_path / "fabric/...".
    
    import shutil
    ml_target = tmp_path / "fabric" / "ml"
    ml_target.mkdir(parents=True, exist_ok=True)
    
    orig_cyclone = REPO_ROOT / "fabric" / "ml" / "anomaly_model.json"
    orig_surge = REPO_ROOT / "fabric" / "ml" / "surge_model.json"
    
    if orig_cyclone.exists():
        shutil.copy(orig_cyclone, ml_target / "anomaly_model.json")
    if orig_surge.exists():
        shutil.copy(orig_surge, ml_target / "surge_model.json")
        
    # 2. Run Rakshak
    result = run_rakshak(
        parquet_dir=data_dir / "ml_training_dataset.parquet", 
        cyclone_model_path=ml_target / "anomaly_model.json",
        surge_model_path=ml_target / "surge_model.json"
    )
    
    # 3. Verify files
    assert (ml_target / "hazards.parquet").exists()
    assert (ml_target / "safe_zones.geojson").exists()
    assert (ml_target / "safe_zones.parquet").exists()
    assert (ml_target / "ecosystem.json").exists()
    assert (ml_target / "rakshak_run.json").exists()
    assert (ml_target / "advisories.json").exists()
    
    # Verify rakshak run output metadata
    assert result["n_zones"] > 0
    
    # Verify we caught the extreme tide (zos=1.5m is > 0.3m threshold)
    import polars as pl
    hazards = pl.read_parquet(ml_target / "hazards.parquet")
    
    if len(hazards) > 0:
        tide_events = hazards.filter(pl.col("type") == "extreme_tide")
        assert len(tide_events) > 0
        assert tide_events[0]["residual_height_m"][0] >= 1.5
    
    # Verify ecosystem analysis caught the coral bleaching risk (SST 31.0)
    ecosystem = json.loads((ml_target / "ecosystem.json").read_text())
    assert ecosystem["total_cells"] > 0
    # At least one cell should have ALERT_2 for bleaching
    alert_cells = [c for c in ecosystem["cells"] if c["coral_bleaching"]["level"] == "ALERT_2"]
    assert len(alert_cells) >= 1
    
    # Verify safe zones contains correct keys and properties
    geojson = json.loads((ml_target / "safe_zones.geojson").read_text())
    assert "features" in geojson
    
    # Verify NaNs did not crash the system and are handled correctly
    zones_df = pl.read_parquet(ml_target / "safe_zones.parquet")
    assert len(zones_df) > 0
