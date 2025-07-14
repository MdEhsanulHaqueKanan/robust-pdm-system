import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import logging
import config
import numpy as np

# --- Setup Professional Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Builds the ColumnTransformer for preprocessing the classification data."""
    categorical_features = ['type']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def train_classification_model():
    """
    Trains the fault classification model and saves the final model and
    preprocessor artifacts.
    """
    logging.info("--- Starting Fault Classification Model Training ---")

    # Load the processed data
    try:
        data = pd.read_csv(config.PROCESSED_DATA_DIR / "classification_processed_data.csv")
        logging.info(f"Loaded processed classification data with shape: {data.shape}")
    except FileNotFoundError:
        logging.error(f"Processed classification data not found at {config.PROCESSED_DATA_DIR}. Aborting.")
        logging.info("Please run the preprocessing script first: python -m scripts.01_preprocess_data")
        return

    # Define features (X) and target (y)
    y = data['failure_type']
    X = data.drop(columns=['target', 'failure_type'])

    # Build and fit the preprocessor
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    logging.info("Features have been preprocessed.")

    # Apply SMOTE to create a balanced dataset for training
    logging.info("Applying SMOTE to create a balanced dataset...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    logging.info(f"Resampling complete. New dataset shape: {X_resampled.shape}")
    
    # Train the LightGBM Classifier
    logging.info("Training LightGBM classifier on resampled data...")
    lgb_clf = lgb.LGBMClassifier(random_state=42)
    lgb_clf.fit(X_resampled, y_resampled)
    logging.info("Model training complete.")

    # Save the model and the preprocessor
    config.CLASSIFICATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = config.CLASSIFICATION_MODEL_DIR / "fault_classifier.joblib"
    preprocessor_path = config.CLASSIFICATION_MODEL_DIR / "classification_preprocessor.pkl"
    
    joblib.dump(lgb_clf, model_path)
    joblib.dump(preprocessor, preprocessor_path) # We save the FITTED preprocessor
    
    logging.info(f"Successfully saved classification model to: {model_path}")
    logging.info(f"Successfully saved classification preprocessor to: {preprocessor_path}")
    logging.info("--- Classification Model Training Script Finished ---")


if __name__ == '__main__':
    train_classification_model()