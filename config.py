from pathlib import Path

# --- Project Root ---
# Define the project root directory to make file paths independent of where the
# project is run from. Path(__file__) is the path to this file (config.py),
# and .parent gives the directory containing it.
PROJECT_ROOT = Path(__file__).parent

# --- Data Paths ---
# Define data-related paths relative to the project root for portability.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific raw data file paths
NASA_DATA_DIR = RAW_DATA_DIR / "C-MAPSS"
KAGGLE_DATA_FILE = RAW_DATA_DIR / "ai4i2020.csv"

# --- Model Artifacts Path ---
# Directory to store trained models, preprocessors, and other ML artifacts.
MODEL_DIR = PROJECT_ROOT / "ml_models"
RUL_MODEL_DIR = MODEL_DIR / "rul_model"
CLASSIFICATION_MODEL_DIR = MODEL_DIR / "classification_model"

# --- RUL Model Config ---
# Rolling window size for feature engineering. 5 cycles was chosen as a balance
# between capturing recent trends and not being overly sensitive to noise.
RUL_WINDOW_SIZE = 5

# Remaining Useful Life (RUL) values are capped at this value. This is a common
# practice in literature for this dataset, as the model's primary goal is to
# predict failure accurately as it becomes imminent, not to distinguish between
# very healthy states (e.g., RUL 200 vs RUL 250).
RUL_CAP = 125

# List of columns to drop from the RUL dataset, identified during EDA as having
# zero or near-zero variance, making them useless for prediction.
RUL_COLS_TO_DROP = [
    'op_setting_3', 'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19'
]

# --- Classification Model Config ---
# List of columns to drop from the classification dataset. These are identifiers
# and not predictive features.
CLASSIFICATION_COLS_TO_DROP = ['udi', 'product_id']