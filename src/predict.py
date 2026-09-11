import os
import joblib
import pandas as pd
import numpy as np
from typing import Union, Dict, Any


def ensure_models_dir(models_dir: str = "models"):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)


def save_model_pipeline(
    pipeline: Any,
    feature_names: list,
    model_name: str = "Best_Model",
    models_dir: str = "models"
) -> str:

    ensure_models_dir(models_dir)
    save_payload = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "model_name": model_name
    }
    file_path = os.path.join(models_dir, "prediction_pipeline.joblib")
    joblib.dump(save_payload, file_path)
    print(f"[Serialization] Saved best model ({model_name}) pipeline to {file_path}")
    return file_path


def load_model_pipeline(pipeline_path: str = "models/prediction_pipeline.joblib") -> dict:

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Saved model pipeline not found at {pipeline_path}")
    payload = joblib.load(pipeline_path)
    return payload


def normalize_input_data(raw_data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:

    if isinstance(raw_data, dict):
        df_input = pd.DataFrame([raw_data])
    else:
        df_input = raw_data.copy()

    # Column mapping dictionary
    col_map = {
        'age': 'Age',
        'gender': 'Gender',
        'total_bilirubin': 'Total_Bilirubin',
        'direct_bilirubin': 'Direct_Bilirubin',
        'alkaline_phosphatase': 'Alkaline_Phosphotase',
        'alkaline_phosphotase': 'Alkaline_Phosphotase',
        'alt': 'Alamine_Aminotransferase',
        'alamine_aminotransferase': 'Alamine_Aminotransferase',
        'ast': 'Aspartate_Aminotransferase',
        'aspartate_aminotransferase': 'Aspartate_Aminotransferase',
        'total_proteins': 'Total_Protiens',
        'total_protiens': 'Total_Protiens',
        'albumin': 'Albumin',
        'ag_ratio': 'Albumin_and_Globulin_Ratio',
        'albumin_and_globulin_ratio': 'Albumin_and_Globulin_Ratio'
    }

    # Rename matching columns
    new_cols = {}
    for col in df_input.columns:
        clean_col = col.strip().lower()
        if clean_col in col_map:
            new_cols[col] = col_map[clean_col]

    df_input = df_input.rename(columns=new_cols)
    return df_input


def predict_liver_disease(
    patient_data: Union[Dict[str, Any], pd.DataFrame],
    pipeline_path: str = "models/prediction_pipeline.joblib"
) -> Dict[str, Any]:
    
    payload = load_model_pipeline(pipeline_path)
    pipeline = payload["pipeline"]
    expected_features = payload["feature_names"]

    df_normalized = normalize_input_data(patient_data)

    if 'Gender' in df_normalized.columns:
        if df_normalized['Gender'].dtype == 'object':
            gender_series = df_normalized['Gender'].astype(str).str.strip().str.capitalize()
            df_normalized['Gender'] = gender_series.map({'Female': 0, 'Male': 1, '0': 0, '1': 1}).fillna(1).astype(int)

    if 'Albumin_and_Globulin_Ratio' in df_normalized.columns and df_normalized['Albumin_and_Globulin_Ratio'].isnull().any():
        df_normalized['Albumin_and_Globulin_Ratio'] = df_normalized['Albumin_and_Globulin_Ratio'].fillna(0.93)

    missing_cols = [col for col in expected_features if col not in df_normalized.columns]
    if missing_cols:
        raise ValueError(f"Missing required clinical input fields: {missing_cols}")

    df_features = df_normalized[expected_features]

    pred_class = int(pipeline.predict(df_features)[0])

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(df_features)[0]
        disease_prob = float(probabilities[1])
    else:
        disease_prob = 1.0 if pred_class == 1 else 0.0

    prob_pct = round(disease_prob * 100, 2)
    outcome_label = "Liver Disease Positive" if pred_class == 1 else "Liver Disease Negative"

    if prob_pct > 70.0:
        risk_category = "High"
    elif prob_pct >= 30.0:
        risk_category = "Moderate"
    else:
        risk_category = "Low"

    return {
        "prediction": outcome_label,
        "prediction_class": pred_class,
        "probability": prob_pct,
        "positive_class_probability_pct": prob_pct,
        "risk_category": risk_category,
        "model_used": payload.get("model_name", "LightGBM Pipeline")
    }


predict_patient_cirrhosis = predict_liver_disease
