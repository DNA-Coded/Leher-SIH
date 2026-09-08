"""
catalog.py — DuckDB Embedded Catalog
Leher / Jal Chakra Pipeline

Reads all manifest.json files from the Samudra Data Fabric and exposes
them as an in-memory DuckDB database for microsecond-speed SQL queries.

No external database server. No PostgreSQL. Just DuckDB + the filesystem.
"""
import duckdb
from pathlib import Path


def init_catalog(fabric_root: Path = Path("fabric")) -> duckdb.DuckDBPyConnection:
    """
    Initialize the DuckDB in-memory catalog from Samudra Data Fabric manifests.

    Loads:
    - All dataset manifests (from fabric/datasets/*/*/manifest.json)
    - Argo float profiles (from fabric/datasets/argo/*/profiles.parquet)
    - ML hazard outputs (from fabric/ml/hazards.parquet if present)

    Returns a connected DuckDB instance ready for SQL queries.
    """
    con = duckdb.connect(":memory:")

    # Load dataset manifests
    manifest_pattern = str(fabric_root / "datasets" / "*" / "*" / "manifest.json")
    manifest_files = list(Path(fabric_root / "datasets").glob("*/*/manifest.json"))

    if manifest_files:
        # DuckDB can read multiple JSON files with a glob pattern
        con.execute(f"""
            CREATE TABLE catalog AS
            SELECT * FROM read_json_auto('{manifest_pattern}', format='auto', filename=true)
        """)
        count = con.execute("SELECT COUNT(*) FROM catalog").fetchone()[0]
        print(f"[CATALOG] Loaded {count} dataset manifest(s)")
    else:
        # Empty catalog — pipeline hasn't run yet
        con.execute("""
            CREATE TABLE catalog (
                dataset_id VARCHAR,
                source_id VARCHAR,
                revision_label VARCHAR,
                is_active BOOLEAN,
                published_at VARCHAR,
                time_min VARCHAR,
                time_max VARCHAR,
                depth_min_m DOUBLE,
                depth_max_m DOUBLE
            )
        """)
        print("[CATALOG] No manifests found — empty catalog initialized")

    # Load Argo profiles if present
    argo_pattern = str(fabric_root / "datasets" / "argo" / "*" / "profiles.parquet")
    argo_files = list(Path(fabric_root / "datasets" / "argo").glob("*/profiles.parquet")) \
        if (fabric_root / "datasets" / "argo").exists() else []

    if argo_files:
        con.execute(f"""
            CREATE TABLE instrument_profiles AS
            SELECT * FROM read_parquet('{argo_pattern}')
        """)
        argo_count = con.execute("SELECT COUNT(*) FROM instrument_profiles").fetchone()[0]
        print(f"[CATALOG] Loaded {argo_count:,} Argo profiles")
    else:
        con.execute("CREATE TABLE instrument_profiles (platform_id VARCHAR, latitude DOUBLE, longitude DOUBLE)")
        print("[CATALOG] No Argo profiles yet")

    # Load ML hazards if present
    hazard_path = fabric_root / "ml" / "hazards.parquet"
    if hazard_path.exists():
        con.execute(f"""
            CREATE TABLE ml_hazards AS
            SELECT * FROM read_parquet('{hazard_path}')
        """)
        h_count = con.execute("SELECT COUNT(*) FROM ml_hazards").fetchone()[0]
        print(f"[CATALOG] Loaded {h_count} hazard record(s)")
    else:
        con.execute("CREATE TABLE ml_hazards (event_id VARCHAR, type VARCHAR, latitude DOUBLE, longitude DOUBLE)")
        print("[CATALOG] No hazards yet — Rakshak inference not run")

    return con


def query_active_dataset(con: duckdb.DuckDBPyConnection, source: str = "glorys") -> dict | None:
    """Return the active dataset manifest for a given source, or None."""
    rows = con.execute("""
        SELECT dataset_id, revision_label, time_min, time_max, depth_min_m, depth_max_m
        FROM catalog
        WHERE is_active = true AND dataset_id LIKE ?
        LIMIT 1
    """, [f"{source}%"]).fetchall()
    if not rows:
        return None
    r = rows[0]
    return {
        "dataset_id": r[0],
        "revision_label": r[1],
        "time_min": r[2],
        "time_max": r[3],
        "depth_min_m": r[4],
        "depth_max_m": r[5],
    }
