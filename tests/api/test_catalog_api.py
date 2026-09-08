import pytest
from fastapi.testclient import TestClient
import json
import numpy as np
import xarray as xr
from pathlib import Path

from apps.api.leher.main import app
from apps.api.leher.main import app

import sys
from pathlib import Path
_jal_chakra_dir = str(Path(__file__).parent.parent.parent / "services" / "jal-chakra")
if _jal_chakra_dir not in sys.path:
    sys.path.insert(0, _jal_chakra_dir)

from core.catalog import init_catalog
from core.samudra_reader import get_active_glorys_dataset

client = TestClient(app)

@pytest.fixture
def mock_fabric(tmp_path):
    fabric_dir = tmp_path / "fabric"
    ds_dir = fabric_dir / "datasets" / "glorys" / "v1"
    ds_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "dataset_id": "glorys_v1",
        "source_id": "GLOBAL_MULTIYEAR",
        "revision_label": "v1",
        "is_active": True,
        "published_at": "2024-01-01T00:00:00Z",
        "time_min": "2024-01-01T00:00:00Z",
        "time_max": "2024-01-02T00:00:00Z",
        "depth_min_m": 0.49,
        "depth_max_m": 500.0,
        "native_depths": [0.49, 1.5, 5.0, 92.3, 109.7, 500.0],
        "bbox": {
            "lon_min": 80.0,
            "lon_max": 80.5,
            "lat_min": 10.0,
            "lat_max": 10.5
        },
        "variables": [
            {"canonical_name": "temperature"},
            {"canonical_name": "salinity"}
        ]
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest))
    
    # Create another inactive one
    ds_dir2 = fabric_dir / "datasets" / "glorys" / "v2"
    ds_dir2.mkdir(parents=True, exist_ok=True)
    manifest2 = dict(manifest)
    manifest2["dataset_id"] = "glorys_v2"
    manifest2["revision_label"] = "v2"
    manifest2["is_active"] = False
    (ds_dir2 / "manifest.json").write_text(json.dumps(manifest2))
    
    lat = [10.0, 10.25, 10.5]
    lon = [80.0, 80.25, 80.5]
    depth = [0.49, 109.7]
    time = np.array(['2024-01-01T00:00:00', '2024-01-02T00:00:00'], dtype='datetime64[ns]')
    
    for var in manifest["variables"]:
        name = var["canonical_name"]
        data = np.ones((2, 2, 3, 3), dtype=np.float32)
        da = xr.DataArray(
            data,
            dims=["time", "depth", "latitude", "longitude"],
            coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon},
            name=name
        )
        ds = da.to_dataset()
        ds.to_zarr(ds_dir / f"{name}.zarr", consolidated=True)
        ds.to_zarr(ds_dir2 / f"{name}.zarr", consolidated=True)
        
    return fabric_dir

@pytest.fixture
def mock_catalog(mock_fabric, monkeypatch):
    monkeypatch.setattr("apps.api.leher.main.REPO_ROOT", mock_fabric.parent)
    monkeypatch.setattr("apps.api.leher.catalog.service.REPO_ROOT", mock_fabric.parent)
    db = init_catalog(mock_fabric)
    app.state.catalog_db = db
    yield db
    db.close()

def test_empty_catalog(tmp_path, monkeypatch):
    fabric_dir = tmp_path / "fabric"
    fabric_dir.mkdir()
    monkeypatch.setattr("apps.api.leher.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("apps.api.leher.catalog.service.REPO_ROOT", tmp_path)
    
    db = init_catalog(fabric_dir)
    app.state.catalog_db = db
    
    response = client.get("/api/v1/catalog/datasets")
    assert response.status_code == 200
    assert response.json() == []
    db.close()

def test_get_datasets(mock_catalog):
    response = client.get("/api/v1/catalog/datasets")
    assert response.status_code == 200
    datasets = response.json()
    assert len(datasets) == 1  # Only the active one
    assert datasets[0]["dataset_id"] == "glorys_v1"

def test_get_dataset_by_id(mock_catalog):
    response = client.get("/api/v1/catalog/datasets/glorys_v1")
    assert response.status_code == 200
    assert response.json()["dataset_id"] == "glorys_v1"
    
    response2 = client.get("/api/v1/catalog/datasets/glorys_v2")
    assert response2.status_code == 200
    assert response2.json()["dataset_id"] == "glorys_v2"

def test_get_dataset_by_id_missing(mock_catalog):
    response = client.get("/api/v1/catalog/datasets/unknown")
    assert response.status_code == 404

def test_get_dataset_variables(mock_catalog):
    response = client.get("/api/v1/catalog/datasets/glorys_v1/variables")
    assert response.status_code == 200
    vars_list = response.json()
    assert len(vars_list) == 2
    assert "temperature" in [v["canonical_name"] for v in vars_list]

def test_get_variable_depths(mock_catalog):
    response = client.get("/api/v1/catalog/variables/temperature/depths")
    assert response.status_code == 200
    depths = response.json()
    assert len(depths) == 6
    assert depths[0] == 0.49

def test_get_variable_times(mock_catalog):
    response = client.get("/api/v1/catalog/variables/temperature/times")
    assert response.status_code == 200
    times = response.json()
    assert len(times) == 2
    assert "2024-01-01T00:00:00.000000000" in times

def test_active_dataset_consistency(mock_catalog, mock_fabric):
    # Test that the catalog and samudra_reader agree on the active dataset
    
    # 1. Ask catalog
    response = client.get("/api/v1/catalog/datasets")
    catalog_active_id = response.json()[0]["dataset_id"]
    
    # 2. Ask samudra_reader
    ds_path, manifest = get_active_glorys_dataset(mock_fabric)
    reader_active_id = manifest["dataset_id"]
    
    assert catalog_active_id == reader_active_id == "glorys_v1"
