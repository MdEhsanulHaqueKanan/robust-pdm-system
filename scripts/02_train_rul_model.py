import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import config

# --- Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def train_rul_model():
    """
    Trains the RUL prediction model using the processed data and saves the
    final model and scaler artifacts.
    """
    logging.info("--- Starting RUL Model Training ---")

    # Load the processed data
    try:
        data = pd.read_csv(config.PROCESSED_DATA_DIR / "rul_processed_data.csv")
        logging.info(f"Loaded processed RUL data with shape: {data.shape}")
    except FileNotFoundError:
        logging.error(f"Processed RUL data not found at {config.PROCESSED_DATA_DIR}. Aborting.")
        logging.info("Please run the preprocessing script first: python -m scripts.01_preprocess_data")
        return

    # Define features (X) and target (y)
    y = data['RUL']
    X = data.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
    
    # Scale the features and preserve column names
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    logging.info("Features have been scaled.")

    # Train the XGBoost Model
    logging.info("Training XGBoost regressor...")
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )
    
    # We train the final model on ALL available data.
    # A dummy eval set is used to enable early stopping.
    xgb_reg.fit(X_scaled, y, eval_set=[(X_scaled, y)], verbose=False)
    logging.info(f"Model training complete. Best iteration: {xgb_reg.best_iteration}")

    # Save the model and the scaler artifacts
    config.RUL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = config.RUL_MODEL_DIR / "rul_predictor.joblib"
    scaler_path = config.RUL_MODEL_DIR / "rul_scaler.pkl"
    
    joblib.dump(xgb_reg, model_path)
    joblib.dump(scaler, scaler_path)
    
    logging.info(f"Successfully saved RUL model to: {model_path}")
    logging.info(f"Successfully saved RUL scaler to: {scaler_path}")
    logging.info("--- RUL Model Training Script Finished ---")


if __name__ == '__main__':
    train_rul_model()