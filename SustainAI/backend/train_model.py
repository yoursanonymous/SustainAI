#!/usr/bin/env python3
"""
SustainAI Model Training Script
Based on the new, rigorous Colab notebook.

--- MODIFIED VERSION (UI-Optimized) ---
This script is now MODIFIED to train models *only* using the features
that are available in the 'MLPrediction.js' frontend UI.
This ensures the UI inputs directly control the prediction.
"""
# CELL 1: Imports
import os, time, warnings, json, joblib, sys, platform, shutil
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.stats import loguniform

import lightgbm
from lightgbm import LGBMRegressor, early_stopping
import xgboost
from xgboost import XGBRegressor
import shap

# Note: codecarbon is imported and used, but not a hard crash if missing
try:
    from codecarbon import EmissionsTracker
except ImportError:
    print("CodeCarbon not found. Training will proceed without emissions tracking.")
    EmissionsTracker = None

# --- Constants ---
RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "GlobalWeatherRepository.csv")
SAVE_DIR = os.path.join(BASE_DIR, "artifacts") # This matches app.py

# --- NEW: DEFINE THE LIST OF FEATURES THAT OUR UI ACTUALLY HAS ---
# These are the *only* features the model will be trained on.
# This MUST match the keys in MLPrediction.js's `inputData` state
UI_FEATURES = [
    "latitude",
    "longitude",
    "temperature_celsius",
    "wind_kph",
    "pressure_mb",
    "humidity",
    "visibility_km",
    "hour",
    "month",
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide"
]


def load_data(csv_path):
    """CELL 2: Robust data loader"""
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"FATAL ERROR: Data file not found.")
        print(f"Please download 'GlobalWeatherRepository.csv' and place it in the 'backend' folder.")
        sys.exit(1)

    expected_cols = [
        "country","location_name","latitude","longitude","timezone",
        "last_updated_epoch","last_updated",
        "temperature_celsius","temperature_fahrenheit",
        "condition_text","wind_mph","wind_kph","wind_degree","wind_direction",
        "pressure_mb","pressure_in","precip_mm","precip_in","humidity","cloud",
        "feels_like_celsius","feels_like_fahrenheit",
        "visibility_km","visibility_miles","uv_index","gust_mph","gust_kph",
        "air_quality_Carbon_Monoxide","air_quality_Ozone","air_quality_Nitrogen_dioxide",
        "air_quality_Sulphur_dioxide","air_quality_PM2.5","air_quality_PM10",
        "air_quality_us-epa-index","air_quality_gb-defra-index",
        "sunrise","sunset","moonrise","moonset","moon_phase","moon_illumination",
    ]

    df_try = pd.read_csv(csv_path, low_memory=False) # Corrected variable name
    if (len(df_try.columns) != 41) or ("air_quality_PM2.5" not in df_try.columns):
        df = pd.read_csv(csv_path, header=None, names=expected_cols, low_memory=False) # Corrected variable name
    else:
        df = df_try.copy()

    print(f"Loaded: {df.shape}")
    return df

def feature_engineer(df):
    """
    CELL 3: Cleanup & Feature Engineering
    We only need to engineer features that are in our UI_FEATURES list
    (e.g., 'hour', 'month').
    """
    print("Running feature engineering...")

    # Parse last_updated to get hour and month
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
        d = df["last_updated"]
        df["hour"]  = d.dt.hour
        df["month"] = d.dt.month
    else:
        # Create dummy hour/month if last_updated is missing
        if "hour" not in df.columns: df["hour"] = 12
        if "month" not in df.columns: df["month"] = 6

    # Fix numeric coercion for UI features
    for c in UI_FEATURES:
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="ignore")

    print(f"Columns after FE: {len(df.columns)}")
    return df

def eval_reg(y_true, y_pred):
    """Helper: Evaluate regression"""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    return dict(R2=r2, RMSE=rmse, MAE=mae)

def train_with_cc(model, X, y, project):
    """Helper: Train with CodeCarbon"""
    if EmissionsTracker is None:
        t0 = time.time(); model.fit(X, y); sec = time.time() - t0; kg = 0.0
        return model, kg, sec

    try:
        tracker = EmissionsTracker(project_name=project, measure_power_secs=5,
                                   gpu_ids=None, tracking_mode="machine", log_level="error")
        tracker.start(); t0 = time.time()
        model.fit(X, y)
        sec = time.time() - t0
        kg  = tracker.stop()
    except Exception:
        t0 = time.time(); model.fit(X, y); sec = time.time() - t0; kg = 0.0
    return model, kg, sec

def infer_with_cc(model, X, project):
    """Helper: Infer with CodeCarbon"""
    if EmissionsTracker is None:
        t0 = time.time(); y_pred = model.predict(X); sec = time.time() - t0; kg = 0.0
        return y_pred, kg, sec

    try:
        tracker = EmissionsTracker(project_name=project, measure_power_secs=5,
                                   gpu_ids=None, tracking_mode="machine", log_level="error")
        tracker.start(); t0 = time.time()
        y_pred = model.predict(X)
        sec = time.time() - t0
        kg  = tracker.stop()
    except Exception:
        t0 = time.time(); y_pred = model.predict(X); sec = time.time() - t0; kg = 0.0
    return y_pred, kg, sec


def train_nowcast_models(df):
    """CELL 4, 5, 6: Train Nowcast Models"""
    print("\n--- Starting NOWCAST Training ---")

    # CELL 4: NUMERIC MATRIX + TARGET CHECK
    TARGET = "air_quality_PM2.5"
    assert TARGET in df.columns, "PM2.5 column missing."

    df_num = df.copy()

    # Ensure all required UI features are present, fill with NaN if not
    for f in UI_FEATURES:
        if f not in df_num.columns:
            df_num[f] = np.nan

    # Coerce all UI features to numeric
    for c in UI_FEATURES:
        # Check if column exists before coercing
        if c in df_num.columns and (df_num[c].dtype == "object"):
            df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    y_now_all = pd.to_numeric(df_num[TARGET], errors="coerce")
    mask_y = y_now_all.notna()

    # --- MODIFICATION: Select ONLY the UI_FEATURES ---
    # Ensure all UI features exist in df_num before selecting
    existing_ui_features = [f for f in UI_FEATURES if f in df_num.columns]
    X_now_all = df_num[existing_ui_features]

    # Align X and y based on valid target
    X_now_all, y_now_all = X_now_all.loc[mask_y], y_now_all.loc[mask_y]

    # Impute missing values *within our selected features*
    imp_now = SimpleImputer(strategy="median")
    X_now_all = pd.DataFrame(imp_now.fit_transform(X_now_all),
                             columns=X_now_all.columns, index=y_now_all.index)
    print(f"NOWCAST dataset (UI Features) → X: {X_now_all.shape}, y: {y_now_all.shape}")

    # CELL 5: SPLITS (NOWCAST)
    Xn_train_full, Xn_test, yn_train_full, yn_test = train_test_split(
        X_now_all, y_now_all, test_size=0.20, random_state=RANDOM_STATE
    )
    Xn_train, Xn_val, yn_train, yn_val = train_test_split(
        Xn_train_full, yn_train_full, test_size=0.125, random_state=RANDOM_STATE
    )
    print(f"NOWCAST splits → Train: {Xn_train.shape}, Val: {Xn_val.shape}, Test: {Xn_test.shape}")

    # CELL 6: LightGBM Training
    print("Training Nowcast LGBM...")
    lgb_space = {
        "n_estimators": [800, 1200], "learning_rate": [0.02, 0.04],
        "num_leaves": [63, 127], "max_depth": [-1, 10],
    }
    lgb_search = RandomizedSearchCV(
        LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1), # Added verbosity
        param_distributions=lgb_space, n_iter=3, cv=2, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    lgb_search.fit(Xn_train, yn_train)
    lgb_best = LGBMRegressor(**lgb_search.best_params_, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1) # Added verbosity

    # Train with early stopping
    lgb_best.fit(
        Xn_train, yn_train,
        eval_set=[(Xn_val, yn_val)],
        callbacks=[early_stopping(stopping_rounds=200, verbose=False)]
    )
    y_pred_lgb, _, _ = infer_with_cc(lgb_best, Xn_test, "Nowcast_LGB_Infer")
    lgb_metrics = eval_reg(yn_test, y_pred_lgb)
    print(f"NOWCAST LGB Metrics: {lgb_metrics}")

    # CELL 6: XGBoost Training
    print("Training Nowcast XGBoost...")
    xgb_space = {
        "n_estimators": [900, 1300], "max_depth": [6, 8, 10],
        "learning_rate": [0.02, 0.04], "subsample": [0.8, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1,
            tree_method="hist", eval_metric="rmse"
        ),
        param_distributions=xgb_space, n_iter=3, cv=2, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    xgb_search.fit(Xn_train, yn_train)
    xgb_best = XGBRegressor(
        **xgb_search.best_params_, objective="reg:squarederror", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist", eval_metric="rmse", early_stopping_rounds=200
    )
    # Pass verbose=False directly to fit for XGBoost
    xgb_best.fit(Xn_train, yn_train, eval_set=[(Xn_val, yn_val)], verbose=False)


    y_pred_xgb, _, _ = infer_with_cc(xgb_best, Xn_test, "Nowcast_XGB_Infer")
    xgb_metrics = eval_reg(yn_test, y_pred_xgb)
    print(f"NOWCAST XGB Metrics: {xgb_metrics}")

    # Aggregate results
    nowcast_results = pd.DataFrame([
        dict(Task="Nowcast", Model="LGB (tuned+ES)", **lgb_metrics),
        dict(Task="Nowcast", Model="XGB (tuned+ES)", **xgb_metrics), # Corrected Task name
    ]).sort_values("R2", ascending=False).reset_index(drop=True)

    print("--- NOWCAST Training Done ---")
    return lgb_best, xgb_best, imp_now, Xn_train, nowcast_results


def train_forecast_models(df):
    """CELL 7, 8: Train Forecast Models"""
    print("\n--- Starting FORECAST Training ---")

    # CELL 7: FORECAST TARGET (t+1) + MATRIX
    df_f = df.copy()
    if {"location_name","last_updated"} <= set(df_f.columns):
        df_f = df_f.sort_values(["location_name","last_updated"])
    else:
        keys = [k for k in ["country","location_name","last_updated_epoch"] if k in df_f.columns]
        if keys: df_f = df_f.sort_values(keys)

    df_f["PM25_t_plus_1"] = df_f.groupby("location_name")["air_quality_PM2.5"].shift(-1)

    df_fnum = df_f.copy()

    # Ensure all required UI features are present
    for f in UI_FEATURES:
        if f not in df_fnum.columns:
            df_fnum[f] = np.nan

    # Coerce all UI features to numeric
    for c in UI_FEATURES:
        # Check if column exists before coercing
        if c in df_fnum.columns and (df_fnum[c].dtype == "object"):
            df_fnum[c] = pd.to_numeric(df_fnum[c], errors="coerce")

    df_fnum = df_fnum.dropna(subset=["PM25_t_plus_1"])

    # --- MODIFICATION: Select ONLY the UI_FEATURES ---
    # Ensure all UI features exist in df_fnum before selecting
    existing_ui_features_f = [f for f in UI_FEATURES if f in df_fnum.columns]
    Xf_all = df_fnum[existing_ui_features_f].copy()
    yf_all = df_fnum["PM25_t_plus_1"].copy()

    # Impute
    imp_f = SimpleImputer(strategy="median")
    Xf_all = pd.DataFrame(imp_f.fit_transform(Xf_all), columns=Xf_all.columns, index=yf_all.index)

    # CELL 7: TIME-AWARE SPLIT
    def time_aware_split_per_group(X, y, groups, test_size=0.20, val_size=0.125, seed=RANDOM_STATE): # Added seed parameter
        idx_train, idx_test = [], []
        # Use groups.unique() for potentially faster unique value retrieval
        for g in groups.unique():
            mask = (groups == g)
            idx_g = np.where(mask)[0]
            n = len(idx_g);
            if n == 0: continue
            n_test = max(1, int(round(test_size * n)))
            split_point = n - n_test
            idx_train.extend(idx_g[:split_point])
            idx_test.extend(idx_g[split_point:])

        X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
        # Ensure groups is Series for iloc
        grp_tr = groups.iloc[idx_train] if isinstance(groups, pd.Series) else pd.Series(groups)[idx_train]
        idx_tr2, idx_val = [], []

        # Use groups.unique() here too
        for g in grp_tr.unique():
            mask = (grp_tr == g)
            # Find indices within the *current* X_tr DataFrame
            idx_g_tr = X_tr[mask].index # Get the original index values
            n = len(idx_g_tr)

            if n == 0: continue
            n_val = max(1, int(round(val_size * n)))
            split_point = n - n_val
            # Use original index values for splitting
            idx_tr2.extend(idx_g_tr[:split_point])
            idx_val.extend(idx_g_tr[split_point:])

        # Return DataFrames/Series using loc with the collected indices
        return (X.loc[idx_tr2], X.loc[idx_val], X.iloc[idx_test],
                y.loc[idx_tr2], y.loc[idx_val], y.iloc[idx_test])


    groups = (df_fnum["location_name"]
              if "location_name" in df_fnum.columns
              else pd.Series("all", index=df_fnum.index))

    # Make sure groups is aligned with Xf_all before splitting
    groups = groups.loc[Xf_all.index]

    Xf_train, Xf_val, Xf_test, yf_train, yf_val, yf_test = time_aware_split_per_group(
        Xf_all, yf_all, groups, test_size=0.20, val_size=0.125, seed=RANDOM_STATE
    )
    print(f"FORECAST splits (time-aware) → Train: {Xf_train.shape}, Val: {Xf_val.shape}, Test: {Xf_test.shape}")

    # CELL 8: LightGBM Training
    print("Training Forecast LGBM...")
    lgb_space_f = {
        "n_estimators": [900, 1300], "learning_rate": [0.02, 0.04],
        "num_leaves": [63, 127], "max_depth": [-1, 10],
    }
    lgb_search_f = RandomizedSearchCV(
        LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1), # Added verbosity
        param_distributions=lgb_space_f, n_iter=3, cv=2, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    lgb_search_f.fit(Xf_train, yf_train)
    lgb_best_f = LGBMRegressor(**lgb_search_f.best_params_, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1) # Added verbosity
    lgb_best_f.fit(
        Xf_train, yf_train,
        eval_set=[(Xf_val, yf_val)],
        callbacks=[early_stopping(stopping_rounds=200, verbose=False)]
    )
    y_pred_lgb_f, _, _ = infer_with_cc(lgb_best_f, Xf_test, "Forecast_LGB_Infer")
    lgb_metrics_f = eval_reg(yf_test, y_pred_lgb_f)
    print(f"FORECAST LGB Metrics: {lgb_metrics_f}")

    # CELL 8: XGBoost Training
    print("Training Forecast XGBoost...")
    xgb_space_f = {
        "n_estimators": [900, 1300], "max_depth": [6, 8, 10],
        "learning_rate": [0.02, 0.04], "subsample": [0.8, 1.0],
    }
    xgb_search_f = RandomizedSearchCV(
        XGBRegressor(
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1,
            tree_method="hist", eval_metric="rmse"
        ),
        param_distributions=xgb_space_f, n_iter=3, cv=2, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1
    )
    xgb_search_f.fit(Xf_train, yf_train)
    xgb_best_f = XGBRegressor(
        **xgb_search_f.best_params_, objective="reg:squarederror", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist", eval_metric="rmse", early_stopping_rounds=200
    )
    # Pass verbose=False directly to fit for XGBoost
    xgb_best_f.fit(Xf_train, yf_train, eval_set=[(Xf_val, yf_val)], verbose=False)


    y_pred_xgb_f, _, _ = infer_with_cc(xgb_best_f, Xf_test, "Forecast_XGB_Infer")
    xgb_metrics_f = eval_reg(yf_test, y_pred_xgb_f)
    print(f"FORECAST XGB Metrics: {xgb_metrics_f}")

    # Aggregate results
    forecast_results = pd.DataFrame([
        dict(Task="Forecast", Model="LGB (tuned+ES)", **lgb_metrics_f),
        dict(Task="Forecast", Model="XGB (tuned+ES)", **xgb_metrics_f),
    ]).sort_values("R2", ascending=False).reset_index(drop=True)

    print("--- FORECAST Training Done ---")
    return lgb_best_f, xgb_best_f, imp_f, Xf_train, forecast_results


def save_artifacts(now_lgb, now_xgb, fore_lgb, fore_xgb, imp_now, imp_f, Xn, Xf, now_results, fore_results):
    print(f"\n--- Saving Artifacts to '{SAVE_DIR}' ---")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save both candidate models
    joblib.dump(now_lgb,  os.path.join(SAVE_DIR, 'nowcast_lgb.pkl'))
    joblib.dump(now_xgb,  os.path.join(SAVE_DIR, 'nowcast_xgb.pkl'))
    joblib.dump(fore_lgb, os.path.join(SAVE_DIR, 'forecast_lgb.pkl'))
    joblib.dump(fore_xgb, os.path.join(SAVE_DIR, 'forecast_xgb.pkl'))

    # Save imputers
    joblib.dump(imp_now, os.path.join(SAVE_DIR, 'nowcast_imputer.pkl'))
    joblib.dump(imp_f,   os.path.join(SAVE_DIR, 'forecast_imputer.pkl'))

    # Save features
    with open(os.path.join(SAVE_DIR, 'features.json'), 'w') as f:
        json.dump({
            'nowcast_features': list(Xn.columns), # Save columns from the training df
            'forecast_features': list(Xf.columns) # Save columns from the training df
        }, f, indent=2)

    # Save which model is best so app.py can pick it
    best_now_name  = now_results.iloc[0]["Model"]
    best_fore_name = fore_results.iloc[0]["Model"]
    with open(os.path.join(SAVE_DIR, 'model_info.json'), 'w') as f:
        json.dump({
            "python": sys.version, "platform": platform.platform(),
            "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__, "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "random_state": int(RANDOM_STATE),
            "best_nowcast_model": best_now_name,
            "best_forecast_model": best_fore_name
        }, f, indent=2)

    now_results.to_csv(os.path.join(SAVE_DIR, "nowcast_results.csv"), index=False)
    fore_results.to_csv(os.path.join(SAVE_DIR, "forecast_results.csv"), index=False)

    for folder in ["codecarbon", "carbon_logs"]:
        if os.path.isdir(folder):
            try: # Add error handling for zip creation
                shutil.make_archive(os.path.join(SAVE_DIR, "carbon_logs"), "zip", folder)
                print("Zipped CodeCarbon logs.")
            except Exception as e:
                print(f"Could not zip CodeCarbon logs: {e}")
            break

    print("--- Artifact Saving Complete ---")


if __name__ == '__main__':
    print("========= Starting SustainAI Model Training (UI-Optimized) ==========")

    df_raw = load_data(CSV_PATH)
    df_feat = feature_engineer(df_raw)

    now_lgb, now_xgb, imp_now, Xn_train, now_results = train_nowcast_models(df_feat)
    fore_lgb, fore_xgb, imp_f, Xf_train, fore_results = train_forecast_models(df_feat)

    save_artifacts(now_lgb, now_xgb, fore_lgb, fore_xgb,
                   imp_now, imp_f, Xn_train, Xf_train,
                   now_results, fore_results)

    print("========= Model Training Completed Successfully! =========")