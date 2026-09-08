import os
from pathlib import Path
from datetime import datetime
import copernicusmarine

def acquire_glorys_data(
    output_dir: Path,
    start_datetime: str,
    end_datetime: str,
    output_filename: str = None,
    min_lon: float = 20.0,
    max_lon: float = 130.0,
    min_lat: float = -40.0,
    max_lat: float = 30.0,
    min_depth: float = 0.0,
    max_depth: float = 5500.0,
) -> Path:
    """
    UTPATTI Stage: Acquires a bounded subset of GLORYS12 data.
    
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

    phy_output = output_filename.replace(".nc", "_phy.nc")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["thetao", "so", "uo", "vo", "zos", "mlotst", "bottomT", "siconc", "sithick"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        minimum_depth=min_depth,
        maximum_depth=max_depth,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        output_filename=phy_output,
        output_directory=str(output_dir),
        force_download=True,
    )
    
    bio_output = output_filename.replace(".nc", "_bio.nc")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
        variables=["chl"],
        minimum_longitude=min_lon,
        maximum_longitude=max_lon,
        minimum_latitude=min_lat,
        maximum_latitude=max_lat,
        minimum_depth=min_depth,
        maximum_depth=max_depth,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        output_filename=bio_output,
        output_directory=str(output_dir),
        force_download=True,
    )
    
    phy_path = output_dir / phy_output
    bio_path = output_dir / bio_output
    
    if not phy_path.exists() or not bio_path.exists():
        raise RuntimeError(f"Copernicus download failed to produce data.")
        
    import xarray as xr
    # Open both and interpolate bio (0.25 deg) to phy (0.083 deg) grid
    ds_phy = xr.open_dataset(phy_path)
    ds_bio = xr.open_dataset(bio_path)
    
    # Interp bio to phy grid to align coordinates
    ds_bio_interp = ds_bio.interp(
        latitude=ds_phy.latitude,
        longitude=ds_phy.longitude,
        depth=ds_phy.depth,
        time=ds_phy.time,
        method="nearest"
    )
    
    ds_merged = xr.merge([ds_phy, ds_bio_interp])
    
    for var in ds_merged.data_vars:
        if var in ds_phy:
            ds_merged[var].encoding = ds_phy[var].encoding
        elif var in ds_bio:
            ds_merged[var].encoding = ds_bio[var].encoding
            
    # Need to preserve CF conventions
    if "Conventions" not in ds_merged.attrs:
        ds_merged.attrs["Conventions"] = ds_phy.attrs.get("Conventions", "CF-1.8")
        
    ds_merged.to_netcdf(output_path)
    
    # Cleanup intermediate files
    ds_phy.close()
    ds_bio.close()
    ds_merged.close()
    phy_path.unlink()
    bio_path.unlink()
        
    return output_path
