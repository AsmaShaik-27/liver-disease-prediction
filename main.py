

import sys
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from src.preprocessing import load_raw_data, get_dataset_info, clean_data, split_data
from src.eda import run_eda
from src.model_training import train_candidate_models, tune_hyperparameters
from src.evaluation import (
    evaluate_all_models, plot_confusion_matrices, plot_roc_curves,
    plot_feature_importance
)
from src.predict import save_model_pipeline, predict_patient_cirrhosis


def print_section_header(title: str):
    """Utility function to print styled terminal section banners."""
    width = 80
    print("\n" + "=" * width)
    print(f" {title.upper()} ".center(width, "="))
    print("=" * width + "\n")


def main():
    print_section_header("Indian Liver Patient - Cirrhosis Prediction ML System")

    print_section_header("1. Dataset Loading & Inspection")
    raw_df = load_raw_data("indian_liver_patient.csv")
    info = get_dataset_info(raw_df)

    print(f"Dataset File Loaded: 'indian_liver_patient.csv'")
    print(f"Dataset Shape: {info['shape'][0]} rows, {info['shape'][1]} columns\n")

    print("Column Names & Data Types:")
    for col, dtype in info['dtypes'].items():
        print(f"  - {col:<30}: {dtype}")

    print("\nFirst 5 Records:")
    print(raw_df.head())

    print("\nTarget Column Analysis:")
    print("  - Target Column Name: 'Dataset'")
    print("  - Value Counts (Raw):")
    print(raw_df['Dataset'].value_counts())
    print("\nTarget Explanation:")
    print("  - Class '1': Patients diagnosed with liver disease / cirrhosis outcome (Positive Class).")
    print("  - Class '2': Non-liver disease patients / healthy control group (Negative Class).")


    print_section_header("2. Data Preprocessing & Cleaning")
    print(f"Missing Values Check (Raw):")
    for col, count in info['missing_values'].items():
        if count > 0:
            print(f"  - {col}: {count} missing value(s)")

    print(f"Duplicate Records Check (Raw): {info['duplicate_count']} duplicated rows found.")

    clean_df = clean_data(raw_df)

    print("\nCategorical and Numerical Features:")
    num_features = [c for c in clean_df.columns if c not in ['Gender', 'Dataset']]
    cat_features = ['Gender']
    print(f"  - Numerical Features ({len(num_features)}): {num_features}")
    print(f"  - Categorical Features ({len(cat_features)}): {cat_features}")
    print(f"  - Target Feature: 'Dataset' (Mapped: 1 = Disease, 0 = Healthy Control)")

    print_section_header("3. Exploratory Data Analysis (EDA)")
    print("Running EDA analysis and generating high-resolution plots in 'plots/' directory...")
    eda_summary = run_eda(clean_df, raw_df=raw_df, plot_dir="plots")

    print("\nEDA Key Findings Summary:")
    print(f"  - Total Cleaned Records: {eda_summary['total_records']}")
    print(f"  - Liver Patient Positive Rate: {eda_summary['positive_class_percentage']:.2f}% ({eda_summary['target_breakdown'][1]} cases)")
    print(f"  - Healthy Control Rate: {100 - eda_summary['positive_class_percentage']:.2f}% ({eda_summary['target_breakdown'][0]} cases)")
    print("\nTop Correlations with Target ('Dataset'):")
    for feature, corr_val in eda_summary['top_positive_correlations'].items():
        print(f"  - {feature:<30}: Correlation = {corr_val:+.4f}")

    print("\nImportant Feature Pairwise Correlations:")
    for f1, f2, val in eda_summary['highly_correlated_pairs']:
        print(f"  - {f1} <-> {f2}: Pearson r = {val:+.4f}")

    print("\nGenerated Plots Saved to 'plots/':")
    print("  1. 01_target_distribution.png")
    print("  2. 02_numerical_distributions.png")
    print("  3. 03_categorical_distribution.png")
    print("  4. 04_correlation_heatmap.png")
    print("  5. 05_feature_boxplots_by_target.png")

    print_section_header("4. Train-Test Split")
    RANDOM_SEED = 42
    X_train, X_test, y_train, y_test = split_data(
        clean_df, target_col='Dataset', test_size=0.2, random_state=RANDOM_SEED
    )
    feature_names = X_train.columns.tolist()

    print_section_header("5. Model Selection & Candidate Training")
    print("Class Imbalance Note: Target ratio is ~71% positive vs ~29% negative.")
    print("Applying SMOTE resampling strictly inside cross-validation & training folds.")

    candidate_pipelines = train_candidate_models(X_train, y_train, random_state=RANDOM_SEED, use_smote=True)

    tuned_pipelines, best_cv_scores = tune_hyperparameters(X_train, y_train, random_state=RANDOM_SEED)

    all_pipelines = {**candidate_pipelines, **tuned_pipelines}

    print_section_header("6. Unseen Test Set Evaluation & Comparison")
    comparison_df, eval_results = evaluate_all_models(all_pipelines, X_test, y_test)

    print("Model Performance Comparison Table (Evaluated on Unseen 20% Test Set):")
    print(comparison_df.to_string(index=False))

    plot_confusion_matrices(eval_results, top_n=4, plot_dir="plots")
    plot_roc_curves(all_pipelines, X_test, y_test, plot_dir="plots")

    print("\nSaved Evaluation Plots to 'plots/':")
    print("  - 06_confusion_matrices.png")
    print("  - 07_roc_curves.png")

    print_section_header("7. Best Model Selection")
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = all_pipelines[best_model_name]
    best_metrics = comparison_df.iloc[0]

    print(f"Selected Best Performing Model: >>> {best_model_name} <<<")
    print("\nBest Model Test Set Metrics:")
    print(f"  - Test Accuracy : {best_metrics['Accuracy'] * 100:.2f}%")
    print(f"  - Precision     : {best_metrics['Precision']:.4f}")
    print(f"  - Recall        : {best_metrics['Recall']:.4f}")
    print(f"  - F1-Score      : {best_metrics['F1-Score']:.4f}")
    print(f"  - ROC-AUC       : {best_metrics['ROC-AUC']:.4f}")

    print("\nRationale for Model Selection:")
    print("  - Medical Diagnosis Priority: Detecting true liver cirrhosis cases (Recall/F1) while maintaining")
    print("    high overall discrimination ability (ROC-AUC) is far more important than raw Accuracy.")
    print(f"  - '{best_model_name}' achieved the highest F1-Score ({best_metrics['F1-Score']:.4f}) and ROC-AUC ({best_metrics['ROC-AUC']:.4f}).")

    saved_model_path = save_model_pipeline(best_pipeline, feature_names, model_name=best_model_name, models_dir="models")

    print_section_header("8. Feature Importance & Explainability")
    feat_imp_df = plot_feature_importance(
        best_pipeline, feature_names, X_test=X_test, y_test=y_test,
        best_model_name=best_model_name, plot_dir="plots"
    )

    print(f"Top Features Influencing Cirrhosis Prediction ({best_model_name}):")
    for idx, row in feat_imp_df.iterrows():
        print(f"  {idx + 1:2d}. {row['Feature']:<30}: {row['Importance']:.4f}")

    print("\nSaved Feature Importance Plot to 'plots/08_feature_importance.png'")

    print_section_header("9. New Patient Prediction Pipeline Verification")
    
    patient_1 = {
        'Age': 62,
        'Gender': 'Male',
        'Total_Bilirubin': 10.9,
        'Direct_Bilirubin': 5.5,
        'Alkaline_Phosphotase': 699,
        'Alamine_Aminotransferase': 64,
        'Aspartate_Aminotransferase': 100,
        'Total_Protiens': 7.5,
        'Albumin': 3.2,
        'Albumin_and_Globulin_Ratio': 0.74
    }

    patient_2 = {
        'Age': 25,
        'Gender': 'Female',
        'Total_Bilirubin': 0.7,
        'Direct_Bilirubin': 0.2,
        'Alkaline_Phosphotase': 160,
        'Alamine_Aminotransferase': 18,
        'Aspartate_Aminotransferase': 15,
        'Total_Protiens': 7.2,
        'Albumin': 4.1,
        'Albumin_and_Globulin_Ratio': 1.30
    }

    print("Simulating Prediction for Patient Case 1 (Severe Biomarker Profile):")
    res1 = predict_patient_cirrhosis(patient_1, pipeline_path=saved_model_path)
    print(f"  - Prediction Class       : {res1['prediction_class']}")
    print(f"  - Outcome Label          : {res1['outcome_label']}")
    print(f"  - Disease Risk           : {res1['disease_probability_pct']}%")
    print(f"  - Clinical Assessment    : {res1['risk_level']}")

    print("\nSimulating Prediction for Patient Case 2 (Normal Biomarker Profile):")
    res2 = predict_patient_cirrhosis(patient_2, pipeline_path=saved_model_path)
    print(f"  - Prediction Class       : {res2['prediction_class']}")
    print(f"  - Outcome Label          : {res2['outcome_label']}")
    print(f"  - Disease Risk           : {res2['disease_probability_pct']}%")
    print(f"  - Clinical Assessment    : {res2['risk_level']}")

    print_section_header("Pipeline Execution Completed Successfully!")


if __name__ == "__main__":
    main()
