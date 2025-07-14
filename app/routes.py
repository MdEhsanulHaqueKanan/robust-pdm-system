from flask import request, jsonify, render_template
from app import app
from app.utils import get_rul_prediction, get_fault_prediction, simulate_drift_detection
import pandas as pd
import json
import numpy as np
import config

# --- Load Data on App Start ---
try:
    rul_df_processed = pd.read_csv(config.PROCESSED_DATA_DIR / "rul_processed_data.csv")
except FileNotFoundError:
    rul_df_processed = None
    print("WARNING: RUL processed data not found. Dashboard and detail pages for RUL assets will be affected.")


@app.route('/')
@app.route('/dashboard')
def dashboard():
    """
    Renders the main dashboard page with DYNAMIC data generated from live model predictions.
    """
    assets_data = []

    # --- Process RUL Assets ---
    if rul_df_processed is not None:
        for unit_id in [1, 4, 20, 50]: 
            asset_info = {"id": f"Turbofan Engine #{unit_id:03}", "type": "rul"}
            unit_history = rul_df_processed[rul_df_processed['unit_number'] == unit_id]
            if not unit_history.empty:
                prediction_result = get_rul_prediction(unit_history)
                asset_info["prediction"] = prediction_result
                rul = prediction_result.get("predicted_rul", 1000)
                if rul < 30:
                    asset_info["status_class"] = "border-red"
                    asset_info["status_text"] = "Critical"
                elif rul < 60:
                    asset_info["status_class"] = "border-orange"
                    asset_info["status_text"] = "Degrading"
                else:
                    asset_info["status_class"] = "border-green"
                    asset_info["status_text"] = "Healthy"
                assets_data.append(asset_info)

    # --- Process Classification Assets ---
    classification_samples = {
        "Milling Machine #XYZ": pd.DataFrame([{ "type": "L", "air_temperature": 302.5, "process_temperature": 311.8, "rotational_speed": 1390, "torque": 55.3, "tool_wear": 208 }]),
        "Conveyor Belt #A-3": pd.DataFrame([{ "type": "M", "air_temperature": 300.1, "process_temperature": 309.7, "rotational_speed": 1550, "torque": 41.2, "tool_wear": 20 }])
    }

    for asset_id, sample_df in classification_samples.items():
        asset_info = {"id": asset_id, "type": "classification"}
        prediction_result = get_fault_prediction(sample_df)
        asset_info["prediction"] = prediction_result
        fault = prediction_result.get("predicted_fault", "No Failure")
        if fault != "No Failure":
            asset_info["status_class"] = "border-red"
            asset_info["status_text"] = "Fault Predicted"
        else:
            asset_info["status_class"] = "border-green"
            asset_info["status_text"] = "Healthy"
        assets_data.append(asset_info)

    return render_template('dashboard.html', title='Dashboard', assets=assets_data)


@app.route('/asset/<asset_id>')
def asset_detail(asset_id):
    """
    Renders the detail page for a specific asset, loading REAL data.
    """
    if 'Turbofan' in asset_id:
        asset_type = 'rul'
        try:
            unit_number = int(asset_id.split('#')[-1])
        except (ValueError, IndexError):
            return "Invalid Turbofan ID format", 404
        if rul_df_processed is not None:
            asset_history_df = rul_df_processed[rul_df_processed['unit_number'] == unit_number].copy()
            historical_data_json = asset_history_df.to_dict(orient='list')
        else:
            historical_data_json = {}
    elif 'Machine' in asset_id or 'Conveyor' in asset_id:
        asset_type = 'classification'
        historical_data_json = {}
    else:
        return "Unknown Asset Type", 404

    return render_template(
        'asset_detail.html',
        title=f"Asset: {asset_id}",
        asset_id=asset_id,
        asset_type=asset_type,
        historical_data=historical_data_json
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    The core API endpoint for getting predictions.
    """
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({"error": f"Failed to parse JSON: {e}"}), 400
    prediction_type = data.get('type')
    payload = data.get('data')
    if not prediction_type or not payload:
        return jsonify({"error": "Missing 'type' or 'data' in request."}), 400
    try:
        df = pd.DataFrame(payload)
    except Exception as e:
        return jsonify({"error": f"Failed to create DataFrame from payload: {e}"}), 400
    if prediction_type == 'rul':
        result = get_rul_prediction(df)
    elif prediction_type == 'classification':
        result = get_fault_prediction(df)
    else:
        return jsonify({"error": f"Invalid prediction type: '{prediction_type}'."}), 400
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result)


@app.route('/model-monitor')
def model_monitor():
    """Renders the model monitoring page."""
    monitoring_data = simulate_drift_detection()
    return render_template(
        'model_monitor.html',
        title="Model Monitor",
        monitoring_data=monitoring_data
    )


@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html', title="About")