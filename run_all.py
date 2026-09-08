"""
run_all.py — Master Execution Script for Leher ML Pipeline
Run this file from the Leher/ root directory.

Executes all phases in order:
  1. Feature engineering from the existing parquet dataset
  2. Download IBTrACS cyclone labels and attach to features
  3. Train XGBoost cyclone detection model
  4. Train XGBoost storm surge regressor
  5. Compute fishing zone GeoJSON
  6. Run full Rakshak inference (produces final backend handover outputs)

Usage:
  python run_all.py

  Or run individual phases:
  python run_all.py --phase 1   (features only)
  python run_all.py --phase 3   (train cyclone model only)
  python run_all.py --phase 6   (Rakshak inference only)
"""
import sys
import argparse
import subprocess
from pathlib import Path

PYTHON = sys.executable
BASE   = Path(__file__).parent

PHASE_SCRIPTS = {
    1: BASE / "services/jal-chakra/core/feature_engineering.py",
    2: BASE / "services/jal-chakra/plugins/copernicus/label_builder.py",
    3: BASE / "services/jal-chakra/core/train_cyclone_model.py",
    4: BASE / "services/jal-chakra/core/train_surge_model.py",
    5: BASE / "services/jal-chakra/core/safe_zones.py",
    6: BASE / "services/jal-chakra/core/rakshak.py",
}

PHASE_NAMES = {
    1: "Feature Engineering (extract surface features from parquet)",
    2: "Label Building (download IBTrACS + attach cyclone labels)",
    3: "Cyclone Model Training (XGBoost classifier)",
    4: "Surge Model Training (XGBoost regressor)",
    5: "Safe Zone Computation (GeoJSON)",
    6: "Rakshak Full Inference (backend handover outputs)",
}

EXPECTED_OUTPUTS = {
    1: Path("fabric/ml/features_unlabeled.parquet"),
    2: Path("fabric/ml/cyclone_training_data.parquet"),
    3: Path("fabric/ml/anomaly_model.json"),
    4: Path("fabric/ml/surge_model.json"),
    5: Path("fabric/ml/safe_zones.geojson"),
    6: Path("fabric/ml/rakshak_run.json"),
    7: Path("fabric/ml/advisories.json"),  # Engine 5 Voice output
}


def run_phase(phase: int):
    script = PHASE_SCRIPTS[phase]
    name   = PHASE_NAMES[phase]

    print()
    print("=" * 60)
    print(f"PHASE {phase}: {name}")
    print("=" * 60)

    if not script.exists():
        print(f"ERROR: Script not found: {script}")
        return False

    result = subprocess.run([PYTHON, str(script)], check=False)

    if result.returncode != 0:
        print(f"\nPHASE {phase} FAILED (exit code {result.returncode})")
        return False

    expected = EXPECTED_OUTPUTS.get(phase)
    if expected and not expected.exists():
        print(f"\nWARNING: Expected output not found: {expected}")
        return False

    print(f"\nPHASE {phase} COMPLETE -> {expected}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Leher ML Pipeline Runner")
    parser.add_argument(
        "--phase", type=int, choices=list(PHASE_SCRIPTS.keys()), default=None,
        help="Run a single phase only (1-6). Omit to run all phases."
    )
    parser.add_argument(
        "--from-phase", type=int, choices=list(PHASE_SCRIPTS.keys()), default=1,
        help="Start from a specific phase (resume after interruption)."
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("LEHER — RAKSHAK INTELLIGENCE PIPELINE")
    print("=" * 60)
    print("Checking prerequisites...")

    # Check that the parquet training data exists
    parquet_dir = BASE / "data/ml_training_dataset.parquet"
    if not parquet_dir.exists():
        print(f"ERROR: Training data not found: {parquet_dir}")
        print("  Copy the parquet files into data/ml_training_dataset.parquet/")
        sys.exit(1)

    # Check required packages
    missing_pkgs = []
    for pkg in ["xgboost", "polars", "duckdb", "sklearn"]:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing_pkgs.append(pkg)

    if missing_pkgs:
        print(f"ERROR: Missing packages: {missing_pkgs}")
        print(f"  Run: pip install {' '.join(missing_pkgs)}")
        sys.exit(1)

    print("Prerequisites OK.\n")

    # Single phase
    if args.phase is not None:
        success = run_phase(args.phase)
        sys.exit(0 if success else 1)

    # All phases in order
    phases = range(args.from_phase, max(PHASE_SCRIPTS.keys()) + 1)
    failed = []

    for phase in phases:
        ok = run_phase(phase)
        if not ok:
            failed.append(phase)
            print(f"\nPipeline stopped at Phase {phase}. Fix the error and re-run with --from-phase {phase}")
            break

    print()
    print("=" * 60)
    if not failed:
        print("ALL PHASES COMPLETE")
        print()
        print("Backend handover outputs:")
        for phase, path in EXPECTED_OUTPUTS.items():
            status = "OK" if path.exists() else "MISSING"
            print(f"  [{status}] {path}")
    else:
        print(f"PIPELINE FAILED at Phase {failed[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
