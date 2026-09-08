import os
from pathlib import Path
from datetime import datetime
import copernicusmarine

def acquire_glorys_data(
    output_dir: Path,
    start_datetime: str,
    end_datetime: str,
    output_filename: str = None
) -> Path:
    """
    UTPATTI Stage: Acquires a bounded subset of GLORYS12 data for the Indian Ocean.
    
    This function acts as the safe boundary to the Copernicus Marine API.
    It enforces the geographic and variable constraints defined in LeherGuide.md.
    
    Args:
        output_dir: Where to save the downloaded NetCDF file.
        start_datetime: ISO format string (e.g., '2024-01-01T00:00:00')
        end_datetime: ISO format string (e.g., '2024-01-01T23:59:59')
        output_filename: Name of the file. If None, auto-generates based on date.
        
    Returns:
        Path to the downloaded .nc file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not output_filename:
        # e.g., raw_glorys_20240101.nc
        date_str = start_datetime.split("T")[0].replace("-", "")
        output_filename = f"raw_glorys_{date_str}.nc"
        
    output_path = output_dir / output_filename
    
    # We do NOT hardcode credentials here. copernicusmarine uses environment variables
    # COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD
    # or the ~/.copernicusmarine/.copernicusmarine-credentials file.

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["thetao", "so", "uo", "vo", "zos", "mlotst", "bottomT", "siconc", "sithick"],
        minimum_longitude=20.0,
        maximum_longitude=130.0,
        minimum_latitude=-40.0,
        maximum_latitude=30.0,
        minimum_depth=0.0,
        maximum_depth=5500.0,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        output_filename=output_filename,
        output_directory=str(output_dir),
        force_download=True,
    )
    
    if not output_path.exists():
        raise RuntimeError(f"Copernicus download failed to produce {output_path}")
        
    return output_path
