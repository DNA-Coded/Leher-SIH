import os

API_TITLE = os.getenv("API_TITLE", "Leher Ocean API")
API_VERSION = os.getenv("API_VERSION", "2.0.0")

# Environment mode
ENV = os.getenv("LEHER_ENV", "development")
DEBUG = ENV == "development"

# Copernicus / Jal Chakra Ingestion Configuration
COPERNICUS_DATASET_ID = os.getenv("COPERNICUS_DATASET_ID", "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m")
JAL_CHAKRA_OUTPUT_ROOT = os.getenv("JAL_CHAKRA_OUTPUT_ROOT", "fabric")

# Copernicus Bounding Box (Indian Ocean Default)
IO_LON_MIN = float(os.getenv("IO_LON_MIN", "20.0"))
IO_LON_MAX = float(os.getenv("IO_LON_MAX", "130.0"))
IO_LAT_MIN = float(os.getenv("IO_LAT_MIN", "-40.0"))
IO_LAT_MAX = float(os.getenv("IO_LAT_MAX", "30.0"))

# CORS — Configurable allowed origins
# Comma-separated list of allowed origins; defaults to localhost dev server.
# Set LEHER_ALLOWED_ORIGINS="https://leher.incois.gov.in,http://localhost:5173" in production.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("LEHER_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if o.strip()
]

# Cache TTL (seconds) for slice/vector responses
CACHE_TTL_SECONDS = int(os.getenv("LEHER_CACHE_TTL", "300"))

# Fabric root (resolved at startup via paths.py, but configurable here)
FABRIC_ROOT = os.getenv("LEHER_FABRIC_ROOT", "fabric")

# ML artifact root
ML_ARTIFACT_ROOT = os.getenv("LEHER_ML_ROOT", "fabric/ml")
