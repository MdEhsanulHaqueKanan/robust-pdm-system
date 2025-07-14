from flask import request, jsonify, render_template
from app import app
from app.utils import get_rul_prediction, get_fault_prediction, simulate_drift_detection, get_dashboard_data
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


# REPLACE THE OLD dashboard FUNCTION with this one

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """
    Renders the main dashboard page by fetching data from utilities.
    """
    assets_to_display = get_dashboard_data(rul_df_processed)
    
    return render_template('dashboard.html', title='Dashboard', assets=assets_to_display)


@app.route('/asset/<asset_id>')
def asset_detail(asset_id):
    if 'Turbofan' in asset_id:
        asset_type = 'rul'
        try: unit_number = int(asset_id.split('#')[-1])
        except (ValueError, IndexError): return "Invalid Turbofan ID format", 404
        if rul_df_processed is not None:
            asset_history_df = rul_df_processed[rul_df_processed['unit_number'] == unit_number].copy()
            historical_data_json = asset_history_df.to_dict(orient='list')
        else: historical_data_json = {}
    elif 'Machine' in asset_id or 'Conveyor' in asset_id:
        asset_type, historical_data_json = 'classification', {}
    else: return "Unknown Asset Type", 404
    return render_template('asset_detail.html', title=f"Asset: {asset_id}", asset_id=asset_id, asset_type=asset_type, historical_data=historical_data_json)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try: data = request.get_json()
    except Exception as e: return jsonify({"error": f"Failed to parse JSON: {e}"}), 400
    prediction_type, payload = data.get('type'), data.get('data')
    if not prediction_type or not payload: return jsonify({"error": "Missing 'type' or 'data' in request."}), 400
    try: df = pd.DataFrame(payload)
    except Exception as e: return jsonify({"error": f"Failed to create DataFrame from payload: {e}"}), 400
    if prediction_type == 'rul':
        result = get_rul_prediction(df, with_shap=True)
    elif prediction_type == 'classification':
        result = get_fault_prediction(df)
    else: return jsonify({"error": f"Invalid prediction type: '{prediction_type}'."}), 400
    if "error" in result: return jsonify(result), 500
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