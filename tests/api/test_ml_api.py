import pytest
import pyarrow.ipc as ipc
import io
import json
import polars as pl
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from apps.api.leher.main import app
from apps.api.leher.routers import ml

client = TestClient(app)

@pytest.fixture
def mock_ml_artifacts(tmp_path, monkeypatch):
    # Mock paths
    ml_dir = tmp_path / "ml"
    ml_dir.mkdir()
    
    hazards_file = ml_dir / "hazards.parquet"
    zones_file = ml_dir / "safe_zones.geojson"
    ecosystem_file = ml_dir / "ecosystem.json"
    run_file = ml_dir / "rakshak_run.json"
    
    monkeypatch.setattr(ml, "HAZARDS_PARQUET", hazards_file)
    monkeypatch.setattr(ml, "SAFE_ZONES_GEOJSON", zones_file)
    monkeypatch.setattr(ml, "ECOSYSTEM_JSON", ecosystem_file)
    monkeypatch.setattr(ml, "RAKSHAK_RUN_JSON", run_file)
    
    class ArtifactMocker:
        def write_hazards(self):
            df = pl.DataFrame({
                "event_id": ["CYC-0001", "SURGE-0002"],
                "type": ["cyclone", "storm_surge"],
                "latitude": [10.5, 12.0],
                "longitude": [80.5, 82.0],
                "probability": [0.85, None],
                "surge_height_m": [None, 1.2],
                "sst": [30.1, np.nan] # testing NaN handling
            })
            df.write_parquet(hazards_file)
            
        def write_zones(self):
            zones = {"type": "FeatureCollection", "features": []}
            zones_file.write_text(json.dumps(zones))
            
        def write_ecosystem(self):
            eco = {"summary": {"avg_stress": 0.4}}
            ecosystem_file.write_text(json.dumps(eco))
            
        def write_run(self):
            run_data = {"run_at": "2024-01-01T00:00:00Z", "n_cyclone_events": 1}
            run_file.write_text(json.dumps(run_data))
            
    return ArtifactMocker()

def test_get_events_missing(mock_ml_artifacts):
    response = client.get("/api/v1/ml/events")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()

def test_get_events_present(mock_ml_artifacts):
    mock_ml_artifacts.write_hazards()
    response = client.get("/api/v1/ml/events")
    if response.status_code != 200:
        print(f"FAILED: {response.content}")
    assert response.status_code == 200
    events = response.json()
    assert len(events) == 2
    assert events[0]["event_id"] == "CYC-0001"

def test_get_event_by_id(mock_ml_artifacts):
    mock_ml_artifacts.write_hazards()
    response = client.get("/api/v1/ml/events/SURGE-0002")
    assert response.status_code == 200
    assert response.json()["type"] == "storm_surge"

def test_get_event_by_id_not_found(mock_ml_artifacts):
    mock_ml_artifacts.write_hazards()
    response = client.get("/api/v1/ml/events/NONEXISTENT")
    assert response.status_code == 404

def test_get_zones_missing(mock_ml_artifacts):
    response = client.get("/api/v1/ml/zones")
    assert response.status_code == 503

def test_get_zones_present(mock_ml_artifacts):
    mock_ml_artifacts.write_zones()
    response = client.get("/api/v1/ml/zones")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/geo+json"

def test_get_ecosystem_missing(mock_ml_artifacts):
    response = client.get("/api/v1/ml/ecosystem")
    assert response.status_code == 503

def test_get_ecosystem_present(mock_ml_artifacts):
    mock_ml_artifacts.write_ecosystem()
    response = client.get("/api/v1/ml/ecosystem")
    assert response.status_code == 200
    assert response.json()["summary"]["avg_stress"] == 0.4

def test_get_run_status(mock_ml_artifacts):
    # Test execution status + missing run metadata
    response = client.get("/api/v1/ml/run-status")
    assert response.status_code == 200
    status = response.json()
    assert status["execution"]["status"] in ["idle", "running", "failed"]
    assert status["last_completed_run"] is None

    # Test with run metadata
    mock_ml_artifacts.write_run()
    response = client.get("/api/v1/ml/run-status")
    assert response.status_code == 200
    status = response.json()
    assert status["last_completed_run"]["n_cyclone_events"] == 1

def test_get_hazards_arrow_missing(mock_ml_artifacts):
    response = client.get("/api/v1/ml/hazards")
    assert response.status_code == 503

def test_get_hazards_arrow_round_trip(mock_ml_artifacts):
    mock_ml_artifacts.write_hazards()
    response = client.get("/api/v1/ml/hazards")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/vnd.apache.arrow.stream"
    
    # Decode IPC stream
    reader = ipc.RecordBatchStreamReader(io.BytesIO(response.content))
    table = reader.read_all()
    df = table.to_pandas()
    
    assert len(df) == 2
    assert "event_id" in df.columns
    assert "sst" in df.columns
    
    # Verify missing values
    assert df["sst"].isna().any()
    assert df["probability"].isna().any()
    
    # Row assertions
    row = df[df["event_id"] == "CYC-0001"].iloc[0]
    assert row["latitude"] == 10.5
    assert row["type"] == "cyclone"
