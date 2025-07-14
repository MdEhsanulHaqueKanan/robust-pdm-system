import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import config # Import our configuration file

def train_rul_model():
    """
    Trains the RUL prediction model and saves the model and scaler artifacts.
    """
    print("--- Starting RUL Model Training ---")

    # Load the processed data
    try:
        data = pd.read_csv(config.PROCESSED_DATA_DIR / "rul_processed_data.csv")
        print(f"Loaded processed RUL data with shape: {data.shape}")
    except FileNotFoundError:
        print("Error: Processed RUL data not found.")
        print("Please run the preprocessing script first: python -m scripts.01_preprocess_data")
        return

    # Define features (X) and target (y)
    y = data['RUL']
    X = data.drop(columns=['unit_number', 'time_in_cycles', 'RUL'])
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Scale the features and put them back into a DataFrame to preserve column names
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("Features have been scaled and put back into a DataFrame.")
    # --- END OF CORRECTION ---

    # Train the XGBoost Model (using the best parameters from our notebook)
    print("Training XGBoost model...")
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
    # We will use a dummy eval set just to enable early stopping.
    # The model is trained on X_scaled, which is now a DataFrame with feature names.
    xgb_reg.fit(X_scaled, y, eval_set=[(X_scaled, y)], verbose=False)
    print("Model training complete.")

    # Save the model and the scaler
    config.RUL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = config.RUL_MODEL_DIR / "rul_predictor.joblib"
    scaler_path = config.RUL_MODEL_DIR / "rul_scaler.pkl"
    
    joblib.dump(xgb_reg, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Successfully saved RUL model to: {model_path}")
    print(f"Successfully saved RUL scaler to: {scaler_path}")

if __name__ == '__main__':
    train_rul_model()
    print("\nRUL model training script finished.")