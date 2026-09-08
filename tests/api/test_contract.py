"""
test_contract.py — API contract, CORS, and hardening tests.

Validates that the backend meets the contract required by the frontend.
"""
import pytest
import json
import io
import numpy as np
import xarray as xr
from pathlib import Path
from fastapi.testclient import TestClient

from apps.api.leher.main import app, scheduler
from apps.api.leher.routers import ml

client = TestClient(app)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_fabric_for_contract(tmp_path, monkeypatch):
    """Minimal Samudra fabric fixture with one active dataset."""
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
            "lon_min": 80.0, "lon_max": 80.5,
            "lat_min": 10.0, "lat_max": 10.5
        },
        "variables": [
            {"canonical_name": "temperature"},
            {"canonical_name": "salinity"},
            {"canonical_name": "u_velocity"},
            {"canonical_name": "v_velocity"},
            {"canonical_name": "current_speed"},
        ]
    }
    (ds_dir / "manifest.json").write_text(json.dumps(manifest))

    lat = [10.0, 10.25, 10.5]
    lon = [80.0, 80.25, 80.5]
    depth = [0.49, 109.7]
    time_vals = np.array(['2024-01-01T00:00:00', '2024-01-02T00:00:00'], dtype='datetime64[ns]')

    for var_info in manifest["variables"]:
        name = var_info["canonical_name"]
        data = np.ones((2, 2, 3, 3), dtype=np.float32)
        if name == "temperature":
            data[1, 0, 0, 0] = np.nan
        da = xr.DataArray(
            data,
            dims=["time", "depth", "latitude", "longitude"],
            coords={"time": time_vals, "depth": depth, "latitude": lat, "longitude": lon},
            name=name
        )
        ds = da.to_dataset()
        ds.to_zarr(ds_dir / f"{name}.zarr", consolidated=True)

    monkeypatch.setattr("apps.api.leher.routers.slices.REPO_ROOT", tmp_path)
    monkeypatch.setattr("apps.api.leher.routers.vectors.REPO_ROOT", tmp_path)

    import sys
    _jc = str(Path(__file__).parent.parent / "services" / "jal-chakra")
    if _jc not in sys.path:
        sys.path.insert(0, _jc)
    from core.catalog import init_catalog
    monkeypatch.setattr("apps.api.leher.main.REPO_ROOT", tmp_path)
    monkeypatch.setattr("apps.api.leher.catalog.service.REPO_ROOT", tmp_path)
    db = init_catalog(fabric_dir)
    app.state.catalog_db = db
    yield fabric_dir
    db.close()


@pytest.fixture
def mock_ml(tmp_path, monkeypatch):
    """Minimal ML artifacts fixture."""
    import polars as pl
    ml_dir = tmp_path / "ml"
    ml_dir.mkdir()

    hazards_file = ml_dir / "hazards.parquet"
    zones_file = ml_dir / "safe_zones.geojson"
    eco_file = ml_dir / "ecosystem.json"
    run_file = ml_dir / "rakshak_run.json"

    monkeypatch.setattr(ml, "HAZARDS_PARQUET", hazards_file)
    monkeypatch.setattr(ml, "SAFE_ZONES_GEOJSON", zones_file)
    monkeypatch.setattr(ml, "ECOSYSTEM_JSON", eco_file)
    monkeypatch.setattr(ml, "RAKSHAK_RUN_JSON", run_file)

    # Write artifacts
    df = pl.DataFrame({
        "event_id": ["CYC-001"],
        "type": ["cyclone"],
        "latitude": [12.0],
        "longitude": [82.0],
        "probability": [0.9],
        "surge_height_m": [None],
        "sst": [30.5],
    })
    df.write_parquet(hazards_file)
    zones_file.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    eco_file.write_text(json.dumps({"summary": {"avg_stress": 0.3}}))
    run_file.write_text(json.dumps({"run_at": "2024-01-01T00:00:00Z", "n_cyclone_events": 1}))


# ─── CORS Tests ────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_allowed_origin(self):
        """CORS preflight from configured dev origin should succeed."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_cors_disallowed_origin(self):
        """CORS preflight from unknown origin should not get allow header."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORSMiddleware returns 400 for disallowed origins
        assert response.headers.get("access-control-allow-origin") is None

    def test_cors_exposed_headers(self):
        """Scientific metadata headers must be exposed via CORS."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )
        exposed = response.headers.get("access-control-expose-headers", "")
        for hdr in ["X-Dataset-Id", "X-Selected-Depth", "X-Data-Min", "X-Data-Max"]:
            assert hdr.lower() in exposed.lower(), f"Header {hdr} not exposed"

    def test_cors_no_wildcard(self):
        """Verify CORS does not use wildcard '*' for allow-origin."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:5173"},
        )
        origin = response.headers.get("access-control-allow-origin", "")
        assert origin != "*"


# ─── Health and Status ──────────────────────────────────────────────────────────

class TestHealthStatus:
    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_status_endpoint(self):
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "catalog_initialized" in data
        assert "scheduler_running" in data
        assert "rakshak" in data
        assert "environment" in data
        assert "cache_ttl_seconds" in data
        assert isinstance(data["cache_ttl_seconds"], int)


# ─── Error Response Schema ──────────────────────────────────────────────────────

class TestErrorResponses:
    def test_404_returns_json_detail(self, mock_fabric_for_contract):
        response = client.get("/api/v1/catalog/datasets/nonexistent_dataset_xyz")
        assert response.status_code == 404
        body = response.json()
        assert "detail" in body

    def test_400_returns_json_detail(self, mock_fabric_for_contract):
        response = client.get(
            "/api/v1/model/slices",
            params={
                "variable": "temperature",
                "depth_m": -10.0,
                "min_lat": 10.0, "max_lat": 10.3,
                "min_lon": 80.0, "max_lon": 80.3,
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert "detail" in body

    def test_503_when_ml_missing(self):
        """ML endpoints return 503 (not 500) when artifacts are absent."""
        # Point to a non-existent path
        import apps.api.leher.routers.ml as ml_mod
        from pathlib import Path as P
        original = ml_mod.HAZARDS_PARQUET
        ml_mod.HAZARDS_PARQUET = P("/nonexistent/hazards.parquet")
        try:
            response = client.get("/api/v1/ml/events")
            assert response.status_code == 503
            assert "detail" in response.json()
        finally:
            ml_mod.HAZARDS_PARQUET = original


# ─── Arrow IPC Schema ───────────────────────────────────────────────────────────

class TestArrowSchema:
    def test_scalar_slice_arrow_schema(self, mock_fabric_for_contract):
        """Scalar slice must return latitude, longitude, value columns."""
        import pyarrow.ipc as ipc
        response = client.get(
            "/api/v1/model/slices",
            params={
                "variable": "temperature",
                "depth_m": 0.0,
                "min_lat": 10.0, "max_lat": 10.3,
                "min_lon": 80.0, "max_lon": 80.3,
            },
        )
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/vnd.apache.arrow.stream"

        reader = ipc.RecordBatchStreamReader(io.BytesIO(response.content))
        table = reader.read_all()
        assert set(table.column_names) == {"latitude", "longitude", "value"}

    def test_vector_slice_arrow_schema(self, mock_fabric_for_contract):
        """Vector slice must return latitude, longitude, u_velocity, v_velocity (+optional current_speed)."""
        import pyarrow.ipc as ipc
        response = client.get(
            "/api/v1/vectors/slices",
            params={
                "depth_m": 0.0,
                "min_lat": 10.0, "max_lat": 10.3,
                "min_lon": 80.0, "max_lon": 80.3,
            },
        )
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/vnd.apache.arrow.stream"

        reader = ipc.RecordBatchStreamReader(io.BytesIO(response.content))
        table = reader.read_all()
        cols = set(table.column_names)
        assert {"latitude", "longitude", "u_velocity", "v_velocity"}.issubset(cols)


# ─── Scientific Headers ──────────────────────────────────────────────────────────

class TestScientificHeaders:
    def test_slice_response_headers(self, mock_fabric_for_contract):
        """Model slice responses must include scientific metadata headers."""
        response = client.get(
            "/api/v1/model/slices",
            params={
                "variable": "temperature",
                "depth_m": 0.0,
                "min_lat": 10.0, "max_lat": 10.3,
                "min_lon": 80.0, "max_lon": 80.3,
            },
        )
        assert response.status_code == 200
        required_headers = [
            "X-Dataset-Id", "X-Selected-Variable", "X-Selected-Time",
            "X-Selected-Depth", "X-Slice-Rows", "X-Slice-Cols",
            "X-Data-Min", "X-Data-Max"
        ]
        for hdr in required_headers:
            assert hdr.lower() in {k.lower() for k in response.headers.keys()}, \
                f"Missing header: {hdr}"

    def test_depth_positive_down(self, mock_fabric_for_contract):
        """Selected depth header value must be positive-down (>= 0)."""
        response = client.get(
            "/api/v1/model/slices",
            params={
                "variable": "salinity",
                "depth_m": 100.0,
                "min_lat": 10.0, "max_lat": 10.3,
                "min_lon": 80.0, "max_lon": 80.3,
            },
        )
        assert response.status_code == 200
        depth = float(response.headers["X-Selected-Depth"])
        assert depth >= 0, f"Depth is negative: {depth}"


# ─── Catalog Contract ─────────────────────────────────────────────────────────

class TestCatalogContract:
    def test_datasets_list_structure(self, mock_fabric_for_contract):
        response = client.get("/api/v1/catalog/datasets")
        assert response.status_code == 200
        datasets = response.json()
        assert isinstance(datasets, list)
        assert len(datasets) >= 1
        ds = datasets[0]
        assert "dataset_id" in ds

    def test_variables_list_structure(self, mock_fabric_for_contract):
        response = client.get("/api/v1/catalog/datasets/glorys_v1/variables")
        assert response.status_code == 200
        variables = response.json()
        assert isinstance(variables, list)
        assert len(variables) >= 1


# ─── ML Contract ──────────────────────────────────────────────────────────────

class TestMLContract:
    def test_events_json_structure(self, mock_ml):
        response = client.get("/api/v1/ml/events")
        assert response.status_code == 200
        events = response.json()
        assert isinstance(events, list)
        assert "event_id" in events[0]
        assert "type" in events[0]

    def test_zones_geojson(self, mock_ml):
        response = client.get("/api/v1/ml/zones")
        assert response.status_code == 200
        assert "geo+json" in response.headers["Content-Type"]

    def test_run_status_structure(self, mock_ml):
        response = client.get("/api/v1/ml/run-status")
        assert response.status_code == 200
        status = response.json()
        assert "execution" in status
        assert "last_completed_run" in status


# ─── Scheduler State ─────────────────────────────────────────────────────────

class TestSchedulerState:
    def test_scheduler_has_rakshak_job(self):
        with TestClient(app):
            jobs = scheduler.get_jobs()
            rakshak_jobs = [j for j in jobs if j.id == "rakshak_hourly"]
            assert len(rakshak_jobs) == 1

    def test_no_duplicate_jobs_on_restart(self):
        """Using replace_existing=True prevents duplicate job registration."""
        with TestClient(app):
            jobs = scheduler.get_jobs()
            all_ids = [j.id for j in jobs]
            assert all_ids.count("rakshak_hourly") == 1
