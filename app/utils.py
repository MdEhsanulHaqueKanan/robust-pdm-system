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
import logging # Import the logging module

# --- Setup Professional Logging for this module ---
# This ensures that messages from utils.py are also logged
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Load Model Artifacts ---
# We load all the model components when the application starts.
# This is more efficient than loading them for every prediction.

# Load RUL model and its scaler
try:
    rul_model = joblib.load(config.RUL_MODEL_DIR / "rul_predictor.joblib")
    rul_scaler = joblib.load(config.RUL_MODEL_DIR / "rul_scaler.pkl")
    logging.info("RUL model and scaler loaded successfully.")
except FileNotFoundError:
    rul_model, rul_scaler = None, None
    logging.error("Could not load RUL model/scaler. Please run the training script.")

# Load Classification model and its preprocessor
try:
    fault_classifier = joblib.load(config.CLASSIFICATION_MODEL_DIR / "fault_classifier.joblib")
    classification_preprocessor = joblib.load(config.CLASSIFICATION_MODEL_DIR / "classification_preprocessor.pkl")
    logging.info("Classification model and preprocessor loaded successfully.")
except FileNotFoundError:
    fault_classifier, classification_preprocessor = None, None
    logging.error("Could not load classification model/preprocessor. Please run the training script.")


# --- Refactored SHAP Plot Generation ---

def _generate_shap_plot(model, explainer, processed_data, feature_names):
    """A generic helper to generate a SHAP force plot."""
    shap_values = explainer(processed_data)
    
    # Logic for regression vs. classification to get correct base_value and shap_values
    is_classification = hasattr(model, 'classes_')
    
    if is_classification:
        prediction = model.predict(processed_data)[0]
        class_list = model.classes_.tolist()
        prediction_index = class_list.index(prediction)
        
        base_value = shap_values.base_values[0, prediction_index]
        shap_values_for_plot = shap_values.values[0, :, prediction_index]
        feature_names_clean = [name.split('__')[1] for name in feature_names] # Clean names for classification
    else: # Regression
        base_value = explainer.expected_value
        shap_values_for_plot = shap_values.values[0]
        feature_names_clean = feature_names # Names for regression are already clean (from model.feature_names_in_)

    # Generate the plot
    force_plot = shap.force_plot(
        base_value=base_value,
        shap_values=shap_values_for_plot,
        features=processed_data[0],
        feature_names=feature_names_clean,
        matplotlib=True, show=False, figsize=(20, 5), text_rotation=15
    )
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(force_plot)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


# --- Prediction Functions ---

def get_rul_prediction(data: pd.DataFrame, with_shap: bool = False) -> dict:
    """Gets RUL prediction and optionally a SHAP plot."""
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
    
    predicted_rul = min(float(rul_model.predict(X_scaled)[0]), config.RUL_CAP)
    
    shap_plot_base64 = None
    if with_shap:
        try:
            explainer = shap.TreeExplainer(rul_model)
            shap_plot_base64 = _generate_shap_plot(rul_model, explainer, X_scaled, required_features)
        except Exception as e:
            logging.error(f"Error generating RUL SHAP plot: {e}")
            traceback.print_exc()

    return {"predicted_rul": round(predicted_rul, 2), "shap_plot": shap_plot_base64}


def get_fault_prediction(data: pd.DataFrame) -> dict:
    """Gets fault prediction, confidence, and a SHAP plot."""
    if fault_classifier is None or classification_preprocessor is None:
        return {"error": "Classification model is not loaded."}
        
    X_processed = classification_preprocessor.transform(data)
    prediction = fault_classifier.predict(X_processed)[0]
    probabilities = fault_classifier.predict_proba(X_processed)[0]
    class_index = np.where(fault_classifier.classes_ == prediction)[0][0]
    confidence = probabilities[class_index]
    
    shap_plot_base64 = None
    try:
        explainer = shap.TreeExplainer(fault_classifier)
        feature_names = classification_preprocessor.get_feature_names_out()
        shap_plot_base64 = _generate_shap_plot(fault_classifier, explainer, X_processed, feature_names)
    except Exception as e:
        logging.error(f"Error generating Classification SHAP plot: {e}")
        traceback.print_exc()

    return {"predicted_fault": str(prediction), "confidence": round(float(confidence), 2), "shap_plot": shap_plot_base64}


# --- Model Monitoring Simulation ---

def simulate_drift_detection():
    """
    Simulates monitoring a data stream for concept drift.
    """
    try:
        df = pd.read_csv(config.PROCESSED_DATA_DIR / "drift_simulation_data.csv")
    except FileNotFoundError:
        logging.error("Drift simulation data not found. Please run the generation script.")
        return {"error": "Drift simulation data not found."}
    initial_train_size = 200
    model = SGDClassifier(loss='log_loss', random_state=42)
    X_initial = df.iloc[:initial_train_size][['feature1', 'feature2']]
    y_initial = df.iloc[:initial_train_size]['target']
    model.fit(X_initial, y_initial)
    stream_data = df.iloc[initial_train_size:]
    chunk_size = 50
    time_steps, accuracies, drift_points = [], [], []
    drift_threshold, is_drift_detected = 0.70, False
    for i in range(0, len(stream_data), chunk_size):
        chunk = stream_data.iloc[i:i+chunk_size]
        if chunk.empty: continue
        X_chunk, y_chunk_true = chunk[['feature1', 'feature2']], chunk['target']
        y_chunk_pred = model.predict(X_chunk)
        acc = accuracy_score(y_chunk_true, y_chunk_pred)
        time_steps.append(initial_train_size + i + chunk_size/2)
        accuracies.append(acc)
        if acc < drift_threshold and not is_drift_detected:
            drift_points.append({"time": initial_train_size + i + chunk_size/2, "label": "Concept Drift Detected"})
            is_drift_detected = True
    return {"time_steps": time_steps, "accuracies": accuracies, "drift_points": drift_points, "drift_threshold": drift_threshold}

# --- Dashboard Data Generation ---

def get_dashboard_data(rul_df_processed: pd.DataFrame) -> list:
    """
    Generates the dynamic data for the dashboard, prioritizing assets by risk.
    This function performs live predictions for dashboard display, showing
    Top N RUL assets and all classification assets.
    """
    logging.info("--- Generating Dynamic Dashboard Data (from utils.py) ---") # Corrected from print()
    all_dashboard_assets = [] # This will collect ALL assets (RUL and Classification)

    # --- Process RUL Assets: Get FAST predictions without SHAP ---
    if rul_df_processed is not None:
        all_rul_predictions = []
        for unit_id in rul_df_processed['unit_number'].unique():
            unit_history = rul_df_processed[rul_df_processed['unit_number'] == unit_id]
            if not unit_history.empty:
                prediction_result = get_rul_prediction(unit_history, with_shap=False)
                prediction_result['unit_number'] = unit_id
                all_rul_predictions.append(prediction_result)
        
        # Sort all RUL predictions to find the most critical
        sorted_rul_predictions = sorted(all_rul_predictions, key=lambda x: x['predicted_rul'])
        
        # Take the TOP 4 most critical RUL engines to display
        num_rul_to_display = 4
        for pred in sorted_rul_predictions[:num_rul_to_display]:
            unit_id = pred['unit_number']
            asset_info = {"id": f"Turbofan Engine #{unit_id:03}", "type": "rul"}
            asset_info["prediction"] = pred
            
            # Determine status based on the live RUL prediction
            rul = pred.get("predicted_rul", 1000)
            if rul < 30: asset_info["status_class"], asset_info["status_text"] = "border-red", "Critical"
            elif rul < 60: asset_info["status_class"], asset_info["status_text"] = "border-orange", "Degrading"
            else: asset_info["status_class"], asset_info["status_text"] = "border-green", "Healthy"
            
            all_dashboard_assets.append(asset_info) # Add to the master list

    # --- Process Classification Assets (Add ALL of them) ---
    classification_samples = {
        "Milling Machine #XYZ": pd.DataFrame([{ "type": "L", "air_temperature": 302.5, "process_temperature": 311.8, "rotational_speed": 1390, "torque": 55.3, "tool_wear": 208 }]),
        "Conveyor Belt #A-3": pd.DataFrame([{ "type": "M", "air_temperature": 300.1, "process_temperature": 309.7, "rotational_speed": 1550, "torque": 41.2, "tool_wear": 20 }])
    }

    for asset_id, sample_df in classification_samples.items():
        asset_info = {"id": asset_id, "type": "classification"}
        prediction_result = get_fault_prediction(sample_df)
        asset_info["prediction"] = prediction_result
        
        # Determine status based on the live fault prediction
        fault = prediction_result.get("predicted_fault", "No Failure")
        if fault != "No Failure":
            asset_info["status_class"] = "border-red"
            asset_info["status_text"] = "Fault Predicted"
        else:
            asset_info["status_class"] = "border-green"
            asset_info["status_text"] = "Healthy"
            
        all_dashboard_assets.append(asset_info) # Add to the master list
    
    # Sort the final list to ensure the most critical items (red, then orange) appear first.
    # This sort applies to the combined list of RUL and classification assets.
    status_order = {"border-red": 0, "border-orange": 1, "border-green": 2, "border-yellow": 3}
    final_sorted_assets = sorted(all_dashboard_assets, key=lambda x: status_order.get(x.get('status_class'), 99))
    
    # Return the combined and sorted list. The dashboard HTML will display all of them.
    return final_sorted_assets