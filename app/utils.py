import shap
import matplotlib
matplotlib.use('Agg') # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import joblib
import pandas as pd
import numpy as np
import config # Our project's configuration
import traceback
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

# --- Load Model Artifacts ---
# We load all the model components when the application starts.
# This is more efficient than loading them for every prediction.

# Load RUL model and its scaler
try:
    rul_model = joblib.load(config.RUL_MODEL_DIR / "rul_predictor.joblib")
    rul_scaler = joblib.load(config.RUL_MODEL_DIR / "rul_scaler.pkl")
    print("RUL model and scaler loaded successfully.")
except FileNotFoundError:
    rul_model, rul_scaler = None, None
    print("Could not load RUL model/scaler. Please run the training script.")

# Load Classification model and its preprocessor
try:
    fault_classifier = joblib.load(config.CLASSIFICATION_MODEL_DIR / "fault_classifier.joblib")
    classification_preprocessor = joblib.load(config.CLASSIFICATION_MODEL_DIR / "classification_preprocessor.pkl")
    print("Classification model and preprocessor loaded successfully.")
except FileNotFoundError:
    fault_classifier, classification_preprocessor = None, None
    print("Could not load classification model/preprocessor. Please run the training script.")


def get_rul_prediction(data: pd.DataFrame) -> dict:
    """
    Takes a dataframe of sensor readings for RUL prediction, preprocesses it,
    and returns the predicted RUL.
    """
    if rul_model is None or rul_scaler is None:
        return {"error": "RUL model is not loaded."}
    
    df = data.copy()
    sensor_cols = [col for col in df.columns if 'sensor' in col and '_rolling' not in col]
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=config.RUL_WINDOW_SIZE, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=config.RUL_WINDOW_SIZE, min_periods=1).std()
    
    df.fillna(0, inplace=True)
    last_row = df.iloc[[-1]]
    required_features = rul_model.feature_names_in_
    features_for_model = last_row[required_features]
    X_scaled = rul_scaler.transform(features_for_model)
    predicted_rul = rul_model.predict(X_scaled)[0]
    predicted_rul = min(float(predicted_rul), config.RUL_CAP)
    return {"predicted_rul": round(predicted_rul, 2)}


def get_fault_prediction(data: pd.DataFrame) -> dict:
    """
    Takes a dataframe of sensor readings for fault classification, preprocesses it,
    and returns the predicted fault type, confidence, and a SHAP plot.
    """
    if fault_classifier is None or classification_preprocessor is None:
        return {"error": "Classification model is not loaded."}
    X_processed = classification_preprocessor.transform(data)
    prediction = fault_classifier.predict(X_processed)[0]
    probabilities = fault_classifier.predict_proba(X_processed)[0]
    class_index = np.where(fault_classifier.classes_ == prediction)[0][0]
    confidence = probabilities[class_index]
    shap_plot_base64 = generate_shap_plot_base64(fault_classifier, classification_preprocessor, data)
    return {
        "predicted_fault": str(prediction),
        "confidence": round(float(confidence), 2),
        "shap_plot": shap_plot_base64
    }


def generate_shap_plot_base64(model, preprocessor, data_row: pd.DataFrame):
    """
    Generates a SHAP force plot for a single prediction and returns it as a 
    base64 encoded string, ready to be embedded in HTML.
    """
    try:
        processed_row = preprocessor.transform(data_row)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(processed_row)
        prediction = model.predict(processed_row)[0]
        class_list = model.classes_.tolist()
        prediction_index = class_list.index(prediction)
        feature_names_raw = preprocessor.get_feature_names_out()
        feature_names_clean = [name.split('__')[1] for name in feature_names_raw]
        force_plot = shap.force_plot(
            base_value=shap_values.base_values[0, prediction_index],
            shap_values=shap_values.values[0, :, prediction_index],
            features=processed_row[0],
            feature_names=feature_names_clean,
            matplotlib=True,
            show=False,
            figsize=(20, 5),
            text_rotation=15
        )
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(force_plot)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        print(f"Error in generate_shap_plot_base64: {e}")
        traceback.print_exc()
        return None


def simulate_drift_detection():
    """
    Simulates monitoring a data stream for concept drift and returns
    data ready for plotting.
    """
    try:
        df = pd.read_csv(config.PROCESSED_DATA_DIR / "drift_simulation_data.csv")
    except FileNotFoundError:
        return {"error": "Drift simulation data not found. Please run the generation script."}

    # 1. Initial Training: Train a model on the first stable concept (first 200 samples)
    initial_train_size = 200
    model = SGDClassifier(loss='log_loss', random_state=42)
    X_initial = df.iloc[:initial_train_size][['feature1', 'feature2']]
    y_initial = df.iloc[:initial_train_size]['target']
    model.fit(X_initial, y_initial)

    # 2. Simulate Monitoring Stream: Process the rest of the data in chunks
    stream_data = df.iloc[initial_train_size:]
    chunk_size = 50
    
    time_steps = []
    accuracies = []
    drift_points = []
    
    # Simple Drift Detection: We'll say drift is detected if accuracy drops below a threshold
    drift_threshold = 0.70
    is_drift_detected = False
    
    for i in range(0, len(stream_data), chunk_size):
        chunk = stream_data.iloc[i:i+chunk_size]
        if chunk.empty:
            continue
            
        X_chunk = chunk[['feature1', 'feature2']]
        y_chunk_true = chunk['target']
        
        y_chunk_pred = model.predict(X_chunk)
        acc = accuracy_score(y_chunk_true, y_chunk_pred)
        
        time_steps.append(initial_train_size + i + chunk_size/2)
        accuracies.append(acc)
        
        if acc < drift_threshold and not is_drift_detected:
            drift_points.append({
                "time": initial_train_size + i + chunk_size/2,
                "label": "Concept Drift Detected"
            })
            is_drift_detected = True
            
    # Prepare data for Plotly charts
    monitoring_data = {
        "time_steps": time_steps,
        "accuracies": accuracies,
        "drift_points": drift_points,
        "drift_threshold": drift_threshold
    }
    
    return monitoring_data