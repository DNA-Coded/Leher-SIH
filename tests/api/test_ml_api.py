import pytest
from fastapi.testclient import TestClient
from apps.api.leher.main import app
from apps.api.leher.paths import (
    HAZARDS_PARQUET,
    SAFE_ZONES_GEOJSON,
    ECOSYSTEM_JSON,
    RAKSHAK_RUN_JSON
)

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_events():
    response = client.get("/api/v1/ml/events")
    if HAZARDS_PARQUET.exists():
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    else:
        assert response.status_code == 404

def test_get_event_not_found():
    response = client.get("/api/v1/ml/events/NONEXISTENT-0000")
    assert response.status_code == 404

def test_get_zones():
    response = client.get("/api/v1/ml/zones")
    if SAFE_ZONES_GEOJSON.exists():
        assert response.status_code == 200
        assert "application/geo+json" in response.headers.get("content-type", "")
    else:
        assert response.status_code == 404

def test_get_ecosystem():
    response = client.get("/api/v1/ml/ecosystem")
    if ECOSYSTEM_JSON.exists():
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
    else:
        assert response.status_code == 404

def test_get_run_status():
    response = client.get("/api/v1/ml/run-status")
    if RAKSHAK_RUN_JSON.exists():
        assert response.status_code == 200
        assert isinstance(response.json(), dict)
    else:
        assert response.status_code == 404
