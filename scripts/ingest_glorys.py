#!/usr/bin/env python3
"""
ingest_glorys.py - Command-line interface for the Jal Chakra GLORYS ingestion pipeline.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Ensure we can import from the workspace
REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Insert services explicitly so absolute imports work
jal_chakra_path = REPO_ROOT / "services" / "jal-chakra"
sys.path.insert(0, str(jal_chakra_path))

try:
    from plugins.copernicus.client import acquire_glorys_data
    from core.pipeline import run_jal_chakra
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ingest GLORYS ocean data into Samudra Data Fabric.")
    
    # Time bounds
    parser.add_argument("--start", type=str, required=True, help="Start datetime (ISO format, e.g., 2024-01-01T00:00:00)")
    parser.add_argument("--end", type=str, required=True, help="End datetime (ISO format, e.g., 2024-01-01T23:59:59)")
    
    # Geographic bounds (defaults to Indian Ocean)
    parser.add_argument("--min-lon", type=float, default=20.0, help="Minimum longitude (default 20.0)")
    parser.add_argument("--max-lon", type=float, default=130.0, help="Maximum longitude (default 130.0)")
    parser.add_argument("--min-lat", type=float, default=-40.0, help="Minimum latitude (default -40.0)")
    parser.add_argument("--max-lat", type=float, default=30.0, help="Maximum latitude (default 30.0)")
    
    # Depth bounds
    parser.add_argument("--min-depth", type=float, default=0.0, help="Minimum depth in meters (default 0.0)")
    parser.add_argument("--max-depth", type=float, default=5500.0, help="Maximum depth in meters (default 5500.0)")
    
    # Execution mode
    parser.add_argument("--dry-run", action="store_true", help="Only parse arguments and validate environment.")

    args = parser.parse_args()

    # Create output directory for raw downloads
    output_dir = REPO_ROOT / "fabric" / "raw" / "copernicus" / "glorys"
    
    # Derive revision label from start date
    try:
        dt = datetime.fromisoformat(args.start)
        revision = f"v{dt.strftime('%Y%m%d')}"
        output_filename = f"raw_glorys_{dt.strftime('%Y%m%d')}.nc"
    except ValueError:
        print("Error: --start must be in ISO format (e.g., 2024-01-01T00:00:00)")
        sys.exit(1)

    print(f"Starting GLORYS ingestion pipeline for revision: {revision}")
    print(f"Time Range: {args.start} to {args.end}")
    print(f"Spatial Bounds: Lon [{args.min_lon}, {args.max_lon}], Lat [{args.min_lat}, {args.max_lat}]")
    print(f"Depth Bounds: [{args.min_depth}m, {args.max_depth}m]")
    
    if args.dry_run:
        print("Dry run completed successfully.")
        return
        
    try:
        # Step 1: Acquire data (UTPATTI)
        print("\n=== Stage 1: UTPATTI (Acquisition) ===")
        raw_nc_path = acquire_glorys_data(
            output_dir=output_dir,
            start_datetime=args.start,
            end_datetime=args.end,
            output_filename=output_filename,
            min_lon=args.min_lon,
            max_lon=args.max_lon,
            min_lat=args.min_lat,
            max_lat=args.max_lat,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        print(f"Data acquired successfully: {raw_nc_path}")
        
        # Steps 2-4: Run Pipeline (PARIKSHA, RUPANTAR, PRAKASHAN)
        print("\n=== Stages 2-4: PARIKSHA, RUPANTAR, PRAKASHAN ===")
        manifest = run_jal_chakra(
            nc_path=raw_nc_path,
            source_name="glorys",
            revision=revision
        )
        
        print("\nIngestion completed successfully!")
        print(f"Active dataset ID: {manifest['dataset_id']}")
        
    except Exception as e:
        print(f"\nIngestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
