import json
from app import app # We import the app instance from our app package

def test_classification_api():
    """
    Tests the /api/predict endpoint for a 'classification' type prediction.
    """
    # Flask's test_client creates a virtual client to send requests to the app
    client = app.test_client()

    # This is the same kind of payload our frontend sends
    payload = {
        "type": "classification",
        "data": [
            {
                "type": "L",
                "air_temperature": 301.5,
                "process_temperature": 310.8,
                "rotational_speed": 1422,
                "torque": 49.3,
                "tool_wear": 193
            }
        ]
    }
    
    # Send a POST request to the API endpoint
    response = client.post(
        '/api/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )

    # --- Assertions: Check if the response is correct ---
    
    # 1. Check if the request was successful (HTTP 200 OK)
    assert response.status_code == 200

    # 2. Parse the JSON response data
    data = json.loads(response.data)

    # 3. Check if the response contains the keys we expect
    assert "predicted_fault" in data
    assert "confidence" in data
    assert "shap_plot" in data
    print("\nClassification API test passed.")


def test_rul_api():
    """
    Tests the /api/predict endpoint for a 'rul' type prediction.
    """
    client = app.test_client()

    # For the RUL model, we need to send a history of sensor data
    # We will create a small, fake history for the test
    payload = {
        "type": "rul",
        "data": [
            {"op_setting_1": -0.0007, "op_setting_2": -0.0004, "sensor_2": 641.82, "sensor_3": 1589.70, "sensor_4": 1400.60, "sensor_7": 554.36, "sensor_8": 2388.06, "sensor_9": 9046.19, "sensor_11": 47.47, "sensor_12": 521.66, "sensor_13": 2388.02, "sensor_14": 8138.62, "sensor_15": 8.4195, "sensor_17": 392, "sensor_20": 39.06, "sensor_21": 23.4190},
            {"op_setting_1": 0.0019, "op_setting_2": -0.0003, "sensor_2": 642.15, "sensor_3": 1591.82, "sensor_4": 1403.14, "sensor_7": 553.75, "sensor_8": 2388.04, "sensor_9": 9044.07, "sensor_11": 47.49, "sensor_12": 522.28, "sensor_13": 2388.07, "sensor_14": 8131.49, "sensor_15": 8.4328, "sensor_17": 392, "sensor_20": 39.00, "sensor_21": 23.4236}
        ]
    }
    
    response = client.post(
        '/api/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    # --- Assertions ---
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "predicted_rul" in data
    assert "shap_plot" in data
    print("RUL API test passed.")