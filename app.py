"""
Flask Web Application for Indian Liver Patient Disease Prediction ML System.
Serves web interface and POST /predict JSON API endpoint using existing pre-trained LightGBM pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure local source package is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.predict import predict_liver_disease, load_model_pipeline

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "prediction_pipeline.joblib")


def validate_patient_inputs(data: dict):
    """
    Validates and casts the 10 clinical input parameters.
    Returns (is_valid, cleaned_dict_or_error_message).
    """
    field_mappings = {
        'age': ('Age', int, 1, 120, "Age must be a positive integer between 1 and 120 years."),
        'gender': ('Gender', str, None, None, "Gender must be either 'Male' or 'Female'."),
        'total_bilirubin': ('Total_Bilirubin', float, 0.0, 100.0, "Total Bilirubin must be a non-negative number."),
        'direct_bilirubin': ('Direct_Bilirubin', float, 0.0, 50.0, "Direct Bilirubin must be a non-negative number."),
        'alkaline_phosphatase': ('Alkaline_Phosphotase', float, 0.0, 5000.0, "Alkaline Phosphatase must be a positive number."),
        'alt': ('Alamine_Aminotransferase', float, 0.0, 5000.0, "Alamine Aminotransferase (ALT) must be a positive number."),
        'ast': ('Aspartate_Aminotransferase', float, 0.0, 5000.0, "Aspartate Aminotransferase (AST) must be a positive number."),
        'total_proteins': ('Total_Protiens', float, 0.0, 20.0, "Total Proteins must be a positive number."),
        'albumin': ('Albumin', float, 0.0, 10.0, "Albumin must be a positive number."),
        'ag_ratio': ('Albumin_and_Globulin_Ratio', float, 0.0, 10.0, "Albumin and Globulin Ratio must be a non-negative number.")
    }

    # Alternate key alias dictionary
    alias_dict = {
        'Age': 'age',
        'Gender': 'gender',
        'Total_Bilirubin': 'total_bilirubin',
        'Direct_Bilirubin': 'direct_bilirubin',
        'Alkaline_Phosphotase': 'alkaline_phosphatase',
        'Alkaline_Phosphatase': 'alkaline_phosphatase',
        'Alamine_Aminotransferase': 'alt',
        'ALT': 'alt',
        'Aspartate_Aminotransferase': 'ast',
        'AST': 'ast',
        'Total_Protiens': 'total_proteins',
        'Total_Proteins': 'total_proteins',
        'Albumin': 'albumin',
        'Albumin_and_Globulin_Ratio': 'ag_ratio',
        'AG_Ratio': 'ag_ratio'
    }

    normalized_data = {}
    for key, val in data.items():
        clean_k = key.strip().lower()
        if clean_k in field_mappings:
            normalized_data[clean_k] = val
        elif key in alias_dict:
            normalized_data[alias_dict[key]] = val
        elif clean_k in [alias_dict[k] for k in alias_dict]:
            normalized_data[clean_k] = val

    # Check missing fields
    missing = [k for k in field_mappings if k not in normalized_data or normalized_data[k] is None or str(normalized_data[k]).strip() == '']
    if missing:
        return False, f"Missing required input fields: {', '.join(missing)}"

    cleaned = {}
    for field, (target_name, dtype, min_val, max_val, err_msg) in field_mappings.items():
        raw_val = str(normalized_data[field]).strip()
        
        if field == 'gender':
            val_cap = raw_val.capitalize()
            if val_cap not in ['Male', 'Female', '1', '0']:
                return False, err_msg
            cleaned[target_name] = val_cap
        else:
            try:
                num_val = dtype(raw_val)
            except (ValueError, TypeError):
                return False, f"Invalid numerical value for {target_name}: '{raw_val}'"

            if min_val is not None and num_val < min_val:
                return False, err_msg
            if max_val is not None and num_val > max_val:
                return False, err_msg

            cleaned[target_name] = num_val

    return True, cleaned


@app.route("/")
def index():
    """Renders the main web interface page."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint confirming model pipeline readiness."""
    exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "healthy" if exists else "model_missing",
        "model_path": MODEL_PATH,
        "model_loaded": exists
    }), 200 if exists else 500


@app.route("/predict_form", methods=["POST"])
def predict_form():
    """Handles web interface prediction requests (form or AJAX)."""
    try:
        if request.is_json:
            raw_data = request.get_json()
        else:
            raw_data = request.form.to_dict()

        is_valid, result = validate_patient_inputs(raw_data)
        if not is_valid:
            return jsonify({"status": "error", "error": result}), 400

        pred_res = predict_liver_disease(result, pipeline_path=MODEL_PATH)

        return jsonify({
            "status": "success",
            "prediction": pred_res["prediction"],
            "prediction_class": pred_res["prediction_class"],
            "probability": pred_res["probability"],
            "risk_category": pred_res["risk_category"],
            "model_used": pred_res["model_used"],
            "input_data": result
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "error": f"Prediction pipeline error: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON API endpoint for Liver Disease Prediction.
    Accepts JSON input with clinical biomarkers and returns prediction, probability, and risk_category.
    """
    try:
        raw_data = request.get_json(silent=True)
        if not raw_data:
            return jsonify({"error": "Invalid request. Expected JSON body with patient biomarkers."}), 400

        is_valid, result = validate_patient_inputs(raw_data)
        if not is_valid:
            return jsonify({"error": result}), 400

        pred_res = predict_liver_disease(result, pipeline_path=MODEL_PATH)

        return jsonify({
            "prediction": pred_res["prediction"],
            "probability": pred_res["probability"],
            "risk_category": pred_res["risk_category"]
        }), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    print(f"Starting Flask App for Liver Disease Prediction...")
    print(f"Verifying model path: {MODEL_PATH}")
    if os.path.exists(MODEL_PATH):
        print("Model file verified successfully!")
    else:
        print("WARNING: Model file not found at path! Please ensure 'models/prediction_pipeline.joblib' exists.")

    app.run(host="0.0.0.0", port=5000, debug=True)
