import os

API_TITLE = os.getenv("API_TITLE", "Leher Ocean API")
API_VERSION = os.getenv("API_VERSION", "2.0.0")

# Jal Chakra Ingestion Configuration
COPERNICUS_DATASET_ID = os.getenv("COPERNICUS_DATASET_ID", "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m")
JAL_CHAKRA_OUTPUT_ROOT = os.getenv("JAL_CHAKRA_OUTPUT_ROOT", "fabric")

# Copernicus Bounding Box (Indian Ocean Default)
IO_LON_MIN = float(os.getenv("IO_LON_MIN", "20.0"))
IO_LON_MAX = float(os.getenv("IO_LON_MAX", "130.0"))
IO_LAT_MIN = float(os.getenv("IO_LAT_MIN", "-40.0"))
IO_LAT_MAX = float(os.getenv("IO_LAT_MAX", "30.0"))
