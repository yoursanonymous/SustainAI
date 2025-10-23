from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
import shap
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# --- GLOBAL VARIABLE FOR WEATHER DATA ---
weather_data_df = None
# -------------------------------------------

# Global variables for models and preprocessors
nowcast_model = None
forecast_model = None
nowcast_imputer = None
forecast_imputer = None
nowcast_features = None
forecast_features = None
nowcast_explainer = None
# --- NEW: Global for accuracy ---
model_accuracy_data = None
# -------------------------------

def load_models():
    """Load pre-trained models, preprocessors, data, and accuracy"""
    global nowcast_model, forecast_model, nowcast_imputer, forecast_imputer
    global nowcast_features, forecast_features, nowcast_explainer
    global weather_data_df
    global model_accuracy_data # <-- ADDED

    MODEL_DIR = 'artifacts' # Ensure this is the correct directory

    try:
        # Load Weather Data
        csv_path = os.path.join(MODEL_DIR, '../GlobalWeatherRepository.csv')
        if os.path.exists(csv_path):
            weather_data_df = pd.read_csv(csv_path, low_memory=False)
            if 'location_name' in weather_data_df.columns:
                weather_data_df['location_name_lower'] = weather_data_df['location_name'].astype(str).str.lower()
            if 'last_updated' in weather_data_df.columns:
                weather_data_df['last_updated'] = pd.to_datetime(weather_data_df['last_updated'], errors='coerce')
                weather_data_df = weather_data_df.sort_values(by='last_updated')
            print(f"Loaded GlobalWeatherRepository.csv with {len(weather_data_df)} rows")
        else:
            print(f"Warning: GlobalWeatherRepository.csv not found at {csv_path}. City endpoints may fail.")

        # --- NEW: LOAD ACCURACY DATA ---
        try:
            model_info_path = os.path.join(MODEL_DIR, 'model_info.json')
            nowcast_results_path = os.path.join(MODEL_DIR, 'nowcast_results.csv')
            forecast_results_path = os.path.join(MODEL_DIR, 'forecast_results.csv')

            now_r2 = None
            fore_r2 = None
            best_now_model_name = "N/A"
            best_fore_model_name = "N/A"

            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                    best_now_model_name = model_info.get("best_nowcast_model", "N/A")
                    best_fore_model_name = model_info.get("best_forecast_model", "N/A")

            if os.path.exists(nowcast_results_path):
                now_df = pd.read_csv(nowcast_results_path)
                # Get R2 for the best model (usually the first row after sorting in train_model.py)
                if not now_df.empty:
                    now_r2 = now_df.iloc[0]['R2']


            if os.path.exists(forecast_results_path):
                fore_df = pd.read_csv(forecast_results_path)
                if not fore_df.empty:
                    fore_r2 = fore_df.iloc[0]['R2']


            model_accuracy_data = {
                "best_nowcast_model": best_now_model_name,
                "nowcast_r2": now_r2,
                "best_forecast_model": best_fore_model_name,
                "forecast_r2": fore_r2
            }
            print("Loaded model accuracy data:", model_accuracy_data)
        except Exception as acc_e:
            print(f"Warning: Could not load accuracy data from CSV/JSON files: {acc_e}")
            model_accuracy_data = None # Ensure it's None if loading fails
        # -------------------------------


        # Load models
        # (Using specific names based on common practice, adjust if your train script saves differently)
        xgb_now_path = os.path.join(MODEL_DIR, 'nowcast_xgb.pkl')
        lgb_now_path = os.path.join(MODEL_DIR, 'nowcast_lgb.pkl')
        xgb_fore_path = os.path.join(MODEL_DIR, 'forecast_xgb.pkl')
        lgb_fore_path = os.path.join(MODEL_DIR, 'forecast_lgb.pkl')

        # Load the *best* nowcast model based on info file
        if model_accuracy_data and "XGB" in model_accuracy_data.get("best_nowcast_model", "") and os.path.exists(xgb_now_path):
            nowcast_model = joblib.load(xgb_now_path)
            print("Loaded best nowcast model (XGBoost)")
        elif model_accuracy_data and "LGB" in model_accuracy_data.get("best_nowcast_model", "") and os.path.exists(lgb_now_path):
            nowcast_model = joblib.load(lgb_now_path)
            print("Loaded best nowcast model (LightGBM)")
        elif os.path.exists(xgb_now_path): # Fallback if info is missing
            nowcast_model = joblib.load(xgb_now_path)
            print("Loaded fallback nowcast model (XGBoost)")
        elif os.path.exists(lgb_now_path):
            nowcast_model = joblib.load(lgb_now_path)
            print("Loaded fallback nowcast model (LightGBM)")


        # Load the *best* forecast model
        if model_accuracy_data and "XGB" in model_accuracy_data.get("best_forecast_model", "") and os.path.exists(xgb_fore_path):
            forecast_model = joblib.load(xgb_fore_path)
            print("Loaded best forecast model (XGBoost)")
        elif model_accuracy_data and "LGB" in model_accuracy_data.get("best_forecast_model", "") and os.path.exists(lgb_fore_path):
            forecast_model = joblib.load(lgb_fore_path)
            print("Loaded best forecast model (LightGBM)")
        elif os.path.exists(xgb_fore_path): # Fallback
            forecast_model = joblib.load(xgb_fore_path)
            print("Loaded fallback forecast model (XGBoost)")
        elif os.path.exists(lgb_fore_path):
            forecast_model = joblib.load(lgb_fore_path)
            print("Loaded fallback forecast model (LightGBM)")


        # Load preprocessors
        if os.path.exists(os.path.join(MODEL_DIR, 'nowcast_imputer.pkl')):
            nowcast_imputer = joblib.load(os.path.join(MODEL_DIR, 'nowcast_imputer.pkl'))
            print("Loaded nowcast imputer")

        if os.path.exists(os.path.join(MODEL_DIR, 'forecast_imputer.pkl')):
            forecast_imputer = joblib.load(os.path.join(MODEL_DIR, 'forecast_imputer.pkl'))
            print("Loaded forecast imputer")

        # Load feature lists
        if os.path.exists(os.path.join(MODEL_DIR, 'features.json')):
            with open(os.path.join(MODEL_DIR, 'features.json'), 'r') as f:
                features = json.load(f)
                nowcast_features = features.get('nowcast_features', [])
                forecast_features = features.get('forecast_features', [])
                print("Loaded feature lists")

        # Create SHAP explainer (only if nowcast model loaded)
        if nowcast_model and nowcast_features:
            nowcast_explainer = shap.TreeExplainer(nowcast_model)
            print("Created SHAP explainer for nowcast model")
        else:
            print("Warning: Nowcast model or features missing, SHAP explainer not created.")


        return True
    except Exception as e:
        print(f"!!! CRITICAL ERROR loading models/data: {e}")
        import traceback
        traceback.print_exc()
        return False

# (Keep preprocess_input, get_latest_city_data, and all API endpoints from previous version)
def preprocess_input(data, is_forecast=False):
    """Preprocess input data for prediction"""
    try:
        # If data is already a DataFrame row (Series), convert to DataFrame
        if isinstance(data, pd.Series):
            df = data.to_frame().T
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)


        features = forecast_features if is_forecast else nowcast_features

        if not features:
            print("Error: Features list is empty.")
            return None

        # Align columns - crucial step
        df = df.reindex(columns=features, fill_value=np.nan)

        imputer = forecast_imputer if is_forecast else nowcast_imputer
        if imputer:
            # Imputer expects numpy array, returns numpy array
            imputed_array = imputer.transform(df)
            # Convert back to DataFrame with correct columns
            df_imputed = pd.DataFrame(imputed_array, columns=features, index=df.index)
            return df_imputed
        else:
            print("Warning: Imputer not loaded. Returning data without imputation.")
            return df # Return aligned but potentially non-imputed data

    except Exception as e:
        print(f"Error preprocessing data: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_latest_city_data(city_name_query):
    if weather_data_df is None or 'location_name_lower' not in weather_data_df.columns:
        return None, "Weather dataset not loaded or missing 'location_name_lower' column."

    city_name_lower = city_name_query.lower()
    city_data = weather_data_df[weather_data_df['location_name_lower'] == city_name_lower]

    # Handle common city name variations (add more if needed)
    if city_data.empty:
        if city_name_lower == 'new delhi':
            city_data = weather_data_df[weather_data_df['location_name_lower'] == 'delhi']
        elif city_name_lower == 'delhi':
            city_data = weather_data_df[weather_data_df['location_name_lower'] == 'new delhi']


    if city_data.empty:
        return None, f"City '{city_name_query}' not found."

    # Assuming already sorted by time during load_models
    latest_data_row = city_data.iloc[-1] # Get the most recent row Series
    return latest_data_row, None

@app.route('/api/explain/city', methods=['POST'])
def explain_by_city():
    """Generates SHAP explanation for a specific city using its latest data."""
    print("\n--- Received request for /api/explain/city ---")
    try:
        if nowcast_explainer is None or nowcast_imputer is None:
            print("ERROR: Explainer or imputer not loaded for city explanation.")
            return jsonify({'error': 'SHAP explainer or imputer not loaded'}), 500

        data = request.get_json()
        city_name = data.get('city')
        if not city_name:
            print("ERROR: No city name provided for explanation.")
            return jsonify({'error': 'City name not provided.'}), 400
        print(f"Explaining city: {city_name}")

        # Get latest data for the city
        latest_data_row, error_msg = get_latest_city_data(city_name)
        if error_msg:
            print(f"ERROR: {error_msg}")
            return jsonify({'error': error_msg}), 404
        print("Latest data retrieved for city.")


        # Preprocess this row's data
        print("Preprocessing city data for SHAP...")
        processed_data = preprocess_input(latest_data_row, is_forecast=False) # Pass the Series directly

        if processed_data is None:
            print("ERROR: Preprocessing city data failed.")
            return jsonify({'error': 'Failed to preprocess city data for SHAP.'}), 500
        print("City data preprocessed successfully.")

        # Calculate SHAP values
        print("Calculating SHAP values for city...")
        # explainer expects DataFrame, even if single row
        shap_values = nowcast_explainer.shap_values(processed_data)
        print("SHAP values calculated.")

        # Convert to dictionary and handle float32
        if nowcast_features and len(shap_values) > 0:
            # shap_values is often [values], access the first element
            shap_values_for_instance = shap_values[0] if isinstance(shap_values, list) and len(shap_values) > 0 else shap_values
            # Handle potential shape issues if shap_values is multi-dimensional for single instance
            if len(shap_values_for_instance.shape) > 1 and shap_values_for_instance.shape[0] == 1:
                shap_values_for_instance = shap_values_for_instance[0]

            shap_dict_raw = dict(zip(nowcast_features, shap_values_for_instance))
            shap_dict = {k: float(v) for k, v in shap_dict_raw.items()}
            print("SHAP dictionary created for city.")
        else:
            print("ERROR: Could not create SHAP dictionary for city.")
            shap_dict = {}

        return jsonify({
            'shap_values': shap_dict,
            'city': city_name
        })

    except Exception as e:
        print(f"!!! EXCEPTION in /api/explain/city: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Explanation by city failed: {str(e)}'}), 500

@app.route('/api/predict/city', methods=['POST'])
def predict_by_city():
    """Predicts nowcast AQI for a city by looking up its latest data."""
    try:
        # Check essential components
        if nowcast_model is None or nowcast_imputer is None:
            return jsonify({'error': 'Model or imputer not loaded.'}), 500

        data = request.get_json()
        city_name = data.get('city')
        if not city_name:
            return jsonify({'error': 'City name not provided.'}), 400

        # Get latest data for the city
        latest_data_row, error_msg = get_latest_city_data(city_name)
        if error_msg:
            return jsonify({'error': error_msg}), 404

        # Preprocess this row's data
        processed_data = preprocess_input(latest_data_row, is_forecast=False)
        if processed_data is None:
            return jsonify({'error': 'Failed to preprocess city data.'}), 500

        # Make prediction
        # Ensure processed_data is DataFrame for predict
        prediction = nowcast_model.predict(processed_data)[0]

        return jsonify({
            'prediction': float(prediction),
            'city': city_name,
            # 'data_used': latest_data_row.to_dict() # Optional: Convert Series to dict if sending back
        })

    except Exception as e:
        print(f"!!! EXCEPTION in /api/predict/city: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction by city failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    # Simple health check
    models_status = {
        'nowcast_model': nowcast_model is not None,
        'forecast_model': forecast_model is not None,
        'nowcast_imputer': nowcast_imputer is not None,
        'forecast_imputer': forecast_imputer is not None,
        'nowcast_explainer': nowcast_explainer is not None,
        'weather_data': weather_data_df is not None
    }
    status = 'healthy' if all(models_status.values()) else 'degraded'
    if not models_status['weather_data']: status = 'error_missing_data'

    return jsonify({
        'status': status,
        'components': models_status,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/predict/nowcast', methods=['POST'])
def predict_nowcast():
    # This endpoint remains for the ML Prediction page
    try:
        if nowcast_model is None or nowcast_imputer is None:
            return jsonify({'error': 'Nowcast model or imputer not loaded'}), 500
        data = request.get_json()
        if not data: return jsonify({'error': 'No data provided'}), 400
        processed_data = preprocess_input(data, is_forecast=False)
        if processed_data is None: return jsonify({'error': 'Error preprocessing data'}), 400
        prediction = nowcast_model.predict(processed_data)[0]
        confidence = min(0.95, max(0.7, 0.85 + np.random.normal(0, 0.05))) # Keep mock confidence
        return jsonify({
            'prediction': float(prediction),
            'confidence': float(confidence),
            'model_type': 'nowcast',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"!!! EXCEPTION in /api/predict/nowcast: {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/predict/forecast', methods=['POST'])
def predict_forecast():
    # This endpoint remains for the ML Prediction page
    try:
        if forecast_model is None or forecast_imputer is None:
            return jsonify({'error': 'Forecast model or imputer not loaded'}), 500
        data = request.get_json()
        if not data: return jsonify({'error': 'No data provided'}), 400
        processed_data = preprocess_input(data, is_forecast=True)
        if processed_data is None: return jsonify({'error': 'Error preprocessing data'}), 400
        prediction = forecast_model.predict(processed_data)[0]
        confidence = min(0.9, max(0.6, 0.8 + np.random.normal(0, 0.05))) # Keep mock confidence
        return jsonify({
            'prediction': float(prediction),
            'confidence': float(confidence),
            'model_type': 'forecast',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"!!! EXCEPTION in /api/predict/forecast: {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    # This endpoint remains for the ML Prediction page
    try:
        if nowcast_explainer is None or nowcast_imputer is None:
            return jsonify({'error': 'SHAP explainer or imputer not loaded'}), 500
        data = request.get_json()
        if not data: return jsonify({'error': 'No data provided'}), 400
        processed_data = preprocess_input(data, is_forecast=False)
        if processed_data is None: return jsonify({'error': 'Error preprocessing data for SHAP'}), 400

        shap_values = nowcast_explainer.shap_values(processed_data)

        if nowcast_features and len(shap_values) > 0:
            shap_values_for_instance = shap_values[0] if isinstance(shap_values, list) and len(shap_values) > 0 else shap_values
            if len(shap_values_for_instance.shape) > 1 and shap_values_for_instance.shape[0] == 1:
                shap_values_for_instance = shap_values_for_instance[0]
            shap_dict_raw = dict(zip(nowcast_features, shap_values_for_instance))
            shap_dict = {k: float(v) for k, v in shap_dict_raw.items()}
        else:
            shap_dict = {}

        return jsonify({
            'shap_values': shap_dict,
            'explanation': 'Feature importance for PM2.5 nowcast prediction',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"!!! EXCEPTION in /api/explain: {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Explanation failed: {str(e)}'}), 500

@app.route('/api/carbon-footprint', methods=['GET'])
def get_carbon_footprint():
    """Get carbon footprint information"""
    try:
        carbon_data = {
            'total_emissions_kg': 1.52,
            'energy_consumed_kwh': 5.4,
            'project_duration_months': 2,
            'models_trained': 2,
            'predictions_made': 1000,
            'carbon_per_prediction': 0.000152
        }
        return jsonify(carbon_data)
    except Exception as e:
        print(f"!!! EXCEPTION in /api/carbon-footprint: {str(e)}")
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Failed to get carbon data: {str(e)}'}), 500

# --- NEW: API ENDPOINT FOR MODEL ACCURACY ---
@app.route('/api/model-accuracy', methods=['GET'])
def get_model_accuracy():
    """Returns the R2 scores for the best nowcast and forecast models."""
    global model_accuracy_data # Access the data loaded on startup
    if model_accuracy_data and model_accuracy_data.get('nowcast_r2') is not None:
        # Return only the necessary info, handle potential nulls
        return jsonify({
            "nowcast_r2": model_accuracy_data.get('nowcast_r2'),
            "forecast_r2": model_accuracy_data.get('forecast_r2')
        })
    else:
        # Return default/error if accuracy data wasn't loaded
        print("Warning: Accuracy data not available for /api/model-accuracy")
        return jsonify({
            "nowcast_r2": None, # Indicate data is missing
            "forecast_r2": None
        }), 404 # Use 404 Not Found or 503 Service Unavailable
# ---------------------------------------------

if __name__ == '__main__':
    print("Loading models and data...")
    if load_models():
        print("Models and data loaded successfully!")
    else:
        print("Warning: Some components failed to load. Endpoints might not work.")
    app.run(debug=True, host='0.0.0.0', port=5000)