import pytest
from fastapi.testclient import TestClient
import pyarrow.ipc as ipc
import io
import json
import numpy as np
import xarray as xr
from pathlib import Path
from apps.api.leher.main import app

client = TestClient(app)

@pytest.fixture
def mock_fabric(tmp_path):
    # Same synthetic fabric setup as the reader tests
    fabric_dir = tmp_path / "fabric"
    ds_dir = fabric_dir / "datasets" / "glorys" / "v1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "dataset_id": "glorys_v1",
        "is_active": True,
        "native_depths": [0.49, 1.5, 5.0, 92.3, 109.7, 500.0],
        "variables": [
            {"canonical_name": "temperature"},
            {"canonical_name": "salinity"}
        ]
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest))
    
    lat = [10.0, 10.25, 10.5]
    lon = [80.0, 80.25, 80.5]
    depth = [0.49, 109.7]
    time = np.array(['2024-01-01T00:00:00', '2024-01-02T00:00:00'], dtype='datetime64[ns]')
    
    for var in manifest["variables"]:
        name = var["canonical_name"]
        data = np.ones((2, 2, 3, 3), dtype=np.float32)
        # Inject NaN for the latest time at surface, lat[0], lon[0]
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

def test_valid_slice_request_arrow_round_trip(mock_fabric, monkeypatch):
    from apps.api.leher.paths import REPO_ROOT
    monkeypatch.setattr("apps.api.leher.routers.slices.REPO_ROOT", mock_fabric.parent)
    
    response = client.get(
        "/api/v1/model/slices",
        params={
            "variable": "temperature",
            "depth_m": 0.0,
            "min_lat": 10.0,
            "max_lat": 10.3,
            "min_lon": 80.0,
            "max_lon": 80.3,
            "resolution": 0.25
        }
    )
    
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/vnd.apache.arrow.stream"
    
    # 5. verify selected depth metadata
    assert float(response.headers["X-Selected-Depth"]) == 0.49
    
    # 2. decode the returned Arrow IPC payload
    reader = ipc.RecordBatchStreamReader(io.BytesIO(response.content))
    table = reader.read_all()
    df = table.to_pandas()
    
    # 3. verify the resulting schema
    assert set(df.columns) == {"latitude", "longitude", "value"}
    
    # 4. verify latitude/longitude bounds
    assert df["latitude"].min() >= 10.0
    assert df["latitude"].max() <= 10.3
    assert df["longitude"].min() >= 80.0
    assert df["longitude"].max() <= 80.3
    
    # 6. verify NaN values survive the round trip
    assert df["value"].isna().any()

def test_unsupported_variable(mock_fabric, monkeypatch):
    monkeypatch.setattr("apps.api.leher.routers.slices.REPO_ROOT", mock_fabric.parent)
    response = client.get(
        "/api/v1/model/slices",
        params={
            "variable": "unknown_var",
            "depth_m": 0.0,
            "min_lat": 10.0,
            "max_lat": 10.3,
            "min_lon": 80.0,
            "max_lon": 80.3
        }
    )
    assert response.status_code == 400
    assert "Unsupported variable" in response.json()["detail"]

def test_invalid_depth(mock_fabric, monkeypatch):
    monkeypatch.setattr("apps.api.leher.routers.slices.REPO_ROOT", mock_fabric.parent)
    response = client.get(
        "/api/v1/model/slices",
        params={
            "variable": "temperature",
            "depth_m": -5.0,
            "min_lat": 10.0,
            "max_lat": 10.3,
            "min_lon": 80.0,
            "max_lon": 80.3
        }
    )
    assert response.status_code == 400
    assert "Depth must be >= 0" in response.json()["detail"]

def test_invalid_bounds(mock_fabric, monkeypatch):
    monkeypatch.setattr("apps.api.leher.routers.slices.REPO_ROOT", mock_fabric.parent)
    response = client.get(
        "/api/v1/model/slices",
        params={
            "variable": "temperature",
            "depth_m": 0.0,
            "min_lat": 10.3,
            "max_lat": 10.0,
            "min_lon": 80.0,
            "max_lon": 80.3
        }
    )
    assert response.status_code == 400
    assert "Invalid bounding box" in response.json()["detail"]

def test_missing_active_dataset(tmp_path, monkeypatch):
    # Empty fabric directory
    fabric_dir = tmp_path / "empty_fabric"
    fabric_dir.mkdir()
    monkeypatch.setattr("apps.api.leher.routers.slices.REPO_ROOT", tmp_path)
    
    from apps.api.leher.routers import slices
    slices._slice_cache.clear()
    
    response = client.get(
        "/api/v1/model/slices",
        params={
            "variable": "temperature",
            "depth_m": 500.0,
            "min_lat": 10.0,
            "max_lat": 10.3,
            "min_lon": 80.0,
            "max_lon": 80.3
        }
    )
    if response.status_code != 503:
        print(f"FAILED missing active dataset: {response.status_code} {response.content}")
    assert response.status_code == 503
    assert "No active dataset" in response.json()["detail"].lower() or "no glorys datasets" in response.json()["detail"].lower()
