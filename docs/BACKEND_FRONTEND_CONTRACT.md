# Leher Backend → Frontend Contract

> **Status:** Verified and tested as of Step 13.
> **Base URL:** `http://localhost:8000`
> **API Docs:** `http://localhost:8000/api/docs` (Swagger UI)

---

## Quick Reference

| Group | Path | Method | Status |
|-------|------|--------|--------|
| Health | `/health` | GET | ✅ Verified |
| Status | `/api/v1/status` | GET | ✅ Verified |
| Catalog | `/api/v1/catalog/datasets` | GET | ✅ Verified |
| Catalog | `/api/v1/catalog/datasets/{id}` | GET | ✅ Verified |
| Catalog | `/api/v1/catalog/datasets/{id}/variables` | GET | ✅ Verified |
| Catalog | `/api/v1/catalog/variables/{var_id}/depths` | GET | ✅ Verified |
| Catalog | `/api/v1/catalog/variables/{var_id}/times` | GET | ✅ Verified |
| Model | `/api/v1/model/slices` | GET | ✅ Verified |
| Vectors | `/api/v1/vectors/slices` | GET | ✅ Verified |
| ML | `/api/v1/ml/events` | GET | ✅ Verified |
| ML | `/api/v1/ml/events/{id}` | GET | ✅ Verified |
| ML | `/api/v1/ml/zones` | GET | ✅ Verified |
| ML | `/api/v1/ml/ecosystem` | GET | ✅ Verified |
| ML | `/api/v1/ml/hazards` | GET | ✅ Verified |
| ML | `/api/v1/ml/run-status` | GET | ✅ Verified |
| ML | `/api/v1/ml/run` | POST | ✅ Verified |

### Deferred Endpoints (Not Yet Implemented)

| Path | Reason |
|------|--------|
| `GET /api/v1/model/slices/surface` | Convenience alias — use `/model/slices` with `depth_m=0` |
| `GET /api/v1/model/slices/statistics` | Phase 5 feature |
| `GET /api/v1/model/slices/anomaly` | Requires climatological baseline (Phase 5) |
| `GET /api/v1/vectors/tiles/{z}/{x}/{y}` | MVT tile encoding deferred — use `/vectors/slices` Arrow API |
| `GET /api/v1/vectors/streamlines` | Client-side streamline computation from `/vectors/slices` data |
| `GET /api/v1/instruments/*` | Argo GDAC ingestion not yet implemented (Phase 5) |
| `GET /api/v1/instruments/comparison` | Requires Argo data (Phase 5) |
| `POST /api/v1/ml/voice_advisory` | LLM + TTS pipeline deferred (Phase 4 extension) |
| `GET /api/v1/ml/cyclone_track/{event_id}` | Track prediction deferred (Phase 4 extension) |
| `POST /api/v1/jobs/isosurface` | Async job queue not yet implemented (Phase 5) |
| `GET /api/v1/jobs/{job_id}` | Async job queue not yet implemented (Phase 5) |
| `POST /api/v1/jobs/transect` | Async job queue not yet implemented (Phase 5) |
| `GET /api/v1/export/csv` | Export module not yet implemented (Phase 5) |
| `GET /api/v1/export/netcdf` | Export module not yet implemented (Phase 5) |
| `GET /api/v1/export/geojson` | Export module not yet implemented (Phase 5) |

---

## CORS Configuration

- **Allowed Origins:** Configurable via `LEHER_ALLOWED_ORIGINS` env var
- **Development Default:** `http://localhost:5173`, `http://localhost:3000`
- **Credentials:** Disabled (no cookies/auth tokens cross-origin)
- **Allowed Methods:** `GET`, `POST`, `OPTIONS`
- **Exposed Headers:** `X-Dataset-Id`, `X-Selected-Variable`, `X-Selected-Time`, `X-Selected-Depth`, `X-Slice-Rows`, `X-Slice-Cols`, `X-Data-Min`, `X-Data-Max`, `X-Variables`

> **Important:** No wildcard `*` origin is used. The frontend origin must be in the allowed list.

---

## Endpoint Details

### `GET /health`

Liveness probe. Always returns 200 if the process is running.

```json
{ "status": "ok" }
```

### `GET /api/v1/status`

Operational readiness. Reports scheduler, catalog, and Rakshak state.

```json
{
  "status": "ok",
  "catalog_initialized": true,
  "scheduler_running": true,
  "rakshak": { "status": "idle" },
  "environment": "development",
  "cache_ttl_seconds": 300
}
```

---

### `GET /api/v1/catalog/datasets`

Returns all **active** dataset manifests.

```json
[
  {
    "dataset_id": "glorys_v20240101",
    "source_id": "GLOBAL_MULTIYEAR",
    "revision_label": "v20240101",
    "is_active": true,
    "published_at": "2024-01-01T00:00:00Z",
    "time_min": "2024-01-01T00:00:00Z",
    "time_max": "2024-01-02T00:00:00Z",
    "depth_min_m": 0.49,
    "depth_max_m": 5727.92,
    "native_depths": [0.49, 1.5, ...],
    "variables": [
      { "canonical_name": "temperature" },
      { "canonical_name": "salinity" }
    ]
  }
]
```

### `GET /api/v1/catalog/datasets/{id}`

Full metadata for a single dataset version. Returns 404 if not found.

### `GET /api/v1/catalog/datasets/{id}/variables`

Returns the `variables` array from the manifest. Returns 404 if dataset not found.

### `GET /api/v1/catalog/variables/{var_id}/depths`

Returns an array of native depth levels (positive-down, in meters) from the active dataset that contains this variable.

```json
[0.49, 1.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, ...]
```

### `GET /api/v1/catalog/variables/{var_id}/times`

Returns all available timesteps as ISO 8601 strings.

```json
["2024-01-01T00:00:00.000000000", "2024-01-02T00:00:00.000000000"]
```

---

### `GET /api/v1/model/slices`

Fetch a bounded 2D depth slice as Apache Arrow IPC.

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dataset_id` | string | No | Active dataset | Specific revision ID |
| `variable` | string | **Yes** | — | Canonical variable name (e.g., `temperature`, `salinity`, `u_velocity`) |
| `depth_m` | float | No | `0.0` | Positive-down depth in meters. Resolved to nearest native level. |
| `time` | string | No | `"latest"` | ISO 8601 timestamp or `"latest"` |
| `min_lat` | float | **Yes** | — | South bound of bounding box |
| `max_lat` | float | **Yes** | — | North bound of bounding box |
| `min_lon` | float | **Yes** | — | West bound of bounding box |
| `max_lon` | float | **Yes** | — | East bound of bounding box |
| `resolution` | float | No | `0.25` | Informational; data returns at native resolution |

#### Response

- **Content-Type:** `application/vnd.apache.arrow.stream`
- **Body:** Apache Arrow IPC stream

#### Arrow Schema

| Column | Type | Description |
|--------|------|-------------|
| `latitude` | Float64 | Latitude coordinate |
| `longitude` | Float64 | Longitude coordinate |
| `value` | Float32 | Variable value (NaN preserved) |

#### Response Headers

| Header | Type | Description |
|--------|------|-------------|
| `X-Dataset-Id` | string | Active dataset revision ID |
| `X-Selected-Variable` | string | Canonical variable name |
| `X-Selected-Time` | string | Actual selected timestamp |
| `X-Selected-Depth` | string | Actual native depth selected (positive-down, meters) |
| `X-Slice-Rows` | string | Number of latitude points |
| `X-Slice-Cols` | string | Number of longitude points |
| `X-Data-Min` | string | Minimum value in slice (ignoring NaN) |
| `X-Data-Max` | string | Maximum value in slice (ignoring NaN) |

#### Example Request

```
GET /api/v1/model/slices?variable=temperature&depth_m=0&min_lat=5&max_lat=25&min_lon=70&max_lon=95
```

#### Decoding in JavaScript

```javascript
import { tableFromIPC } from 'apache-arrow';

const response = await fetch('/api/v1/model/slices?...');
const buffer = await response.arrayBuffer();
const table = tableFromIPC(buffer);

const latitudes = table.getChild('latitude').toArray();
const longitudes = table.getChild('longitude').toArray();
const values = table.getChild('value').toArray();

// Read metadata from headers
const depth = response.headers.get('X-Selected-Depth');
const dataMin = response.headers.get('X-Data-Min');
const dataMax = response.headers.get('X-Data-Max');
```

---

### `GET /api/v1/vectors/slices`

Fetch bounded 2D current velocity vectors as Apache Arrow IPC.

#### Query Parameters

Same as `/model/slices` except `variable` is not needed (always returns `u_velocity`, `v_velocity`, and `current_speed` if available).

#### Arrow Schema

| Column | Type | Description |
|--------|------|-------------|
| `latitude` | Float64 | Latitude coordinate |
| `longitude` | Float64 | Longitude coordinate |
| `u_velocity` | Float32 | Eastward current velocity (m/s) |
| `v_velocity` | Float32 | Northward current velocity (m/s) |
| `current_speed` | Float32 | Derived speed (m/s), optional |

#### Response Headers

| Header | Type | Description |
|--------|------|-------------|
| `X-Dataset-Id` | string | Active dataset revision ID |
| `X-Selected-Time` | string | Actual selected timestamp |
| `X-Selected-Depth` | string | Actual native depth (positive-down) |
| `X-Slice-Rows` | string | Number of latitude points |
| `X-Slice-Cols` | string | Number of longitude points |
| `X-Variables` | string | Comma-separated variable names |

---

### `GET /api/v1/ml/events`

Returns all active hazard events detected by Rakshak.

```json
[
  {
    "event_id": "CYC-0001",
    "type": "cyclone",
    "latitude": 12.5,
    "longitude": 82.0,
    "probability": 0.85,
    "surge_height_m": null,
    "sst": 30.1
  }
]
```

- Returns **503** if Rakshak has not completed its first scan.
- `null` values represent missing/not-applicable fields.
- NaN values in source data are converted to `null` in JSON.

### `GET /api/v1/ml/events/{id}`

Returns a single hazard event. Returns 404 if not found, 503 if artifacts absent.

### `GET /api/v1/ml/zones`

Returns GeoJSON FeatureCollection of safe/caution/danger fishing zones.

- **Content-Type:** `application/geo+json`
- Returns **503** if Rakshak has not completed its first scan.

### `GET /api/v1/ml/ecosystem`

Returns ecosystem stress data.

```json
{ "summary": { "avg_stress": 0.4 } }
```

### `GET /api/v1/ml/hazards`

Returns all hazards as Apache Arrow IPC (for high-performance rendering).

- **Content-Type:** `application/vnd.apache.arrow.stream`
- Returns **503** if artifacts absent.

### `GET /api/v1/ml/run-status`

Returns Rakshak execution state and last completed run metadata.

```json
{
  "execution": { "status": "idle" },
  "last_completed_run": {
    "run_at": "2024-01-01T00:00:00Z",
    "n_cyclone_events": 1
  }
}
```

`execution.status` is one of: `idle`, `running`, `failed`.

### `POST /api/v1/ml/run`

Manually triggers a Rakshak scan. Non-blocking; returns immediately.

```json
{ "status": "accepted" }
```

If another scan is already running:
```json
{ "status": "skipped", "reason": "rakshak_run_already_in_progress" }
```

---

## Depth Semantics

- All depths are **positive-down** (0 = surface, 5727.92 = deepest GLORYS level)
- When `depth_m=0` is requested, the API selects the nearest native depth (typically 0.49m)
- The actual depth selected is always returned in `X-Selected-Depth`
- Negative depth values return **400 Bad Request**

## Time Semantics

- `time=latest` returns the most recent timestep in the dataset
- Specific timesteps must match an exact ISO 8601 string from the dataset's time coordinate
- Invalid timestamps return **400 Bad Request**

## NaN Behavior

- **Arrow IPC:** NaN values are preserved as IEEE 754 NaN in Float32/Float64 columns
- **JSON responses:** NaN values are converted to `null` for strict JSON compliance
- NaN is **never** silently converted to zero
- Use `X-Data-Min` / `X-Data-Max` headers to determine the valid data range (NaN-excluded)

## Error Behavior

All errors return JSON with a `detail` field:

```json
{ "detail": "Human-readable error message" }
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Invalid parameters (bad depth, bad bounds, unsupported variable) |
| 404 | Resource not found (unknown dataset, event, variable) |
| 500 | Internal server error (filesystem paths are not exposed) |
| 503 | Service unavailable (ML artifacts not yet generated, catalog not initialized) |

## Caching Behavior

- Slice and vector responses are cached in-memory for 300 seconds (configurable via `LEHER_CACHE_TTL` env var)
- Cache key includes: dataset_id, variable, depth, time, bounding box, resolution
- The cache is process-local and cleared on restart
- The `GET /api/v1/status` endpoint reports the current `cache_ttl_seconds`

## Rakshak Artifact Behavior

- Rakshak runs on an hourly schedule via APScheduler (no Celery)
- ML artifacts are stored at `fabric/ml/` (hazards.parquet, safe_zones.geojson, ecosystem.json)
- If artifacts don't exist, ML endpoints return **503**, not 500
- A manual trigger is available via `POST /api/v1/ml/run`
- Concurrent runs are prevented by a thread lock

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LEHER_ALLOWED_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated CORS allowed origins |
| `LEHER_CACHE_TTL` | `300` | Cache TTL in seconds |
| `LEHER_ENV` | `development` | Environment mode |
| `LEHER_FABRIC_ROOT` | `fabric` | Path to Samudra Data Fabric |
| `LEHER_ML_ROOT` | `fabric/ml` | Path to ML artifacts |
| `COPERNICUS_DATASET_ID` | `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` | Copernicus dataset ID |
| `IO_LON_MIN/MAX, IO_LAT_MIN/MAX` | `20,130,-40,30` | Indian Ocean bounding box |

---

*This contract was verified against 71 passing tests (51 existing + 20 contract tests). Only endpoints listed as ✅ Verified are safe to consume in the frontend.*
