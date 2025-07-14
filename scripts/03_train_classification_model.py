import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import config # Import our configuration file

def train_classification_model():
    """
    Trains the fault classification model and saves the model and preprocessor artifacts.
    """
    print("--- Starting Fault Classification Model Training ---")

    # Load the processed data
    try:
        data = pd.read_csv(config.PROCESSED_DATA_DIR / "classification_processed_data.csv")
        print(f"Loaded processed classification data with shape: {data.shape}")
    except FileNotFoundError:
        print("Error: Processed classification data not found.")
        print("Please run the preprocessing script first: python -m scripts.01_preprocess_data")
        return

    # Define features (X) and target (y)
    y = data['failure_type']
    X = data.drop(columns=['target', 'failure_type'])

    # Define the preprocessing steps using the same logic as the notebook
    categorical_features = ['type']
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor and transform the data
    X_processed = preprocessor.fit_transform(X)
    print("Features have been preprocessed.")

    # Apply SMOTE to the entire processed dataset for final model training
    print("Applying SMOTE to create a balanced dataset for training...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    print(f"Resampling complete. New dataset shape: {X_resampled.shape}")
    
    # Train the LightGBM Classifier
    print("Training LightGBM model on resampled data...")
    lgb_clf = lgb.LGBMClassifier(random_state=42)
    lgb_clf.fit(X_resampled, y_resampled)
    print("Model training complete.")

    # Save the model and the preprocessor
    config.CLASSIFICATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = config.CLASSIFICATION_MODEL_DIR / "fault_classifier.joblib"
    preprocessor_path = config.CLASSIFICATION_MODEL_DIR / "classification_preprocessor.pkl"
    
    joblib.dump(lgb_clf, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    
    print(f"Successfully saved classification model to: {model_path}")
    print(f"Successfully saved classification preprocessor to: {preprocessor_path}")


if __name__ == '__main__':
    train_classification_model()
    print("\nClassification model training script finished.")