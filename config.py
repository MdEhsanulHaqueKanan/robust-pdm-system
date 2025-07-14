from pathlib import Path

# --- Project Root ---
# Define the project root directory.
# Path(__file__) is the path to the current file (config.py).
# .parent gives you the directory containing it (the project root).
PROJECT_ROOT = Path(__file__).parent

# --- Data Paths ---
# Define paths relative to the project root for better portability.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific raw data file paths
NASA_DATA_DIR = RAW_DATA_DIR / "C-MAPSS"
KAGGLE_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"

# --- Model Artifacts Path ---
MODEL_DIR = PROJECT_ROOT / "ml_models"
RUL_MODEL_DIR = MODEL_DIR / "rul_model"
CLASSIFICATION_MODEL_DIR = MODEL_DIR / "classification_model"

# --- RUL Model Config ---
RUL_WINDOW_SIZE = 5
RUL_CAP = 125
RUL_COLS_TO_DROP = [
    'op_setting_3', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19'
]

# --- Classification Model Config ---
CLASSIFICATION_COLS_TO_DROP = ['udi', 'product_id']