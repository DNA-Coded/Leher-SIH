import duckdb
from typing import List, Dict, Any, Optional
import zarr
from apps.api.leher.paths import REPO_ROOT

def _row_to_dict(cursor, row):
    return dict(zip([c[0] for c in cursor.description], row))

def get_active_datasets(db: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    """Returns all active dataset manifests."""
    try:
        # Check if table exists
        db.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'catalog'")
        if not db.fetchone():
            return []
            
        cursor = db.execute("SELECT * FROM catalog WHERE is_active = true")
        rows = cursor.fetchall()
        return [_row_to_dict(cursor, r) for r in rows]
    except Exception as e:
        return []

def get_dataset(db: duckdb.DuckDBPyConnection, dataset_id: str) -> Optional[Dict[str, Any]]:
    """Returns metadata for a single dataset version."""
    try:
        db.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'catalog'")
        if not db.fetchone():
            return None
            
        cursor = db.execute("SELECT * FROM catalog WHERE dataset_id = ?", [dataset_id])
        row = cursor.fetchone()
        if not row:
            return None
        return _row_to_dict(cursor, row)
    except Exception:
        return None

def get_dataset_variables(db: duckdb.DuckDBPyConnection, dataset_id: str) -> Optional[List[Dict[str, Any]]]:
    """Returns the variables available in a dataset."""
    try:
        dataset = get_dataset(db, dataset_id)
        if not dataset:
            return None
        return dataset.get("variables", [])
    except Exception:
        return None

def get_variable_active_dataset(db: duckdb.DuckDBPyConnection, var_id: str) -> Optional[Dict[str, Any]]:
    """Finds the active dataset that provides the given variable ID."""
    try:
        # Fetch all active datasets
        active = get_active_datasets(db)
        for ds in active:
            variables = ds.get("variables", [])
            # Variables is a list of dicts from duckdb JSON struct
            for v in variables:
                if isinstance(v, dict) and v.get("canonical_name") == var_id:
                    return ds
                elif isinstance(v, str) and v == var_id:
                    return ds
        return None
    except Exception:
        return None

def get_variable_times(db: duckdb.DuckDBPyConnection, var_id: str) -> Optional[List[str]]:
    """
    Returns available timesteps for a variable.
    Since manifest only has time_min/time_max, we read the exact coordinate values
    from the Zarr metadata directly without loading data arrays.
    """
    try:
        ds = get_variable_active_dataset(db, var_id)
        if not ds:
            return None
            
        source_id = ds.get("source_id", "glorys") # Default if not matched
        # Our pipeline saves to fabric/datasets/<source>/<revision>
        # Let's derive source from dataset_id (e.g. glorys_v20240101 -> glorys)
        dataset_id = ds.get("dataset_id", "")
        source_name = dataset_id.split("_")[0] if "_" in dataset_id else "glorys"
        revision = ds.get("revision_label", "")
        
        zarr_path = REPO_ROOT / "fabric" / "datasets" / source_name / revision / f"{var_id}.zarr"
        if not zarr_path.exists():
            return None
            
        # Open zarr explicitly for the time array
        # xarray is slow, so we read zarr directly for coords
        z = zarr.open(str(zarr_path), mode='r')
        if "time" not in z:
            return []
            
        # time values in glorys are typically days since 1950, etc.
        # But wait, samudra_reader reads them using xarray for proper cftime decoding.
        # So using xarray directly is safer for time decoding.
        import xarray as xr
        xds = xr.open_zarr(zarr_path, consolidated=True, decode_times=True)
        if "time" not in xds.coords:
            return []
            
        # Convert numpy datetime64 to ISO strings
        times = [str(t) for t in xds.time.values]
        return times
        
    except Exception as e:
        print(f"Error reading times: {e}")
        return None
