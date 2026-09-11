"""
Model Testing, Evaluation, Comparison, Feature Importance, and Plotting Module.
Computes test metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix),
generates comparison tables, plots ROC curves and feature importances.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Tuple, Dict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.inspection import permutation_importance


def ensure_plot_dir(plot_dir: str = "plots"):
    """Creates output plot directory if needed."""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


def evaluate_single_model(pipeline: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluates a single model pipeline on unseen test set.
    """
    y_pred = pipeline.predict(X_test)

    # Get prediction probabilities for ROC-AUC
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_prob = pipeline.decision_function(X_test)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion_Matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def evaluate_all_models(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, dict]:
    """
    Evaluates a dictionary of model pipelines on X_test, y_test.
    Returns:
    - Comparison DataFrame sorted by F1-Score / ROC-AUC.
    - Detailed evaluation dict containing confusion matrices & probabilities.
    """
    eval_results = {}
    table_rows = []

    for name, pipeline in pipelines.items():
        res = evaluate_single_model(pipeline, X_test, y_test)
        eval_results[name] = res
        table_rows.append({
            "Model": name,
            "Accuracy": res["Accuracy"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1-Score": res["F1-Score"],
            "ROC-AUC": res["ROC-AUC"]
        })

    comparison_df = pd.DataFrame(table_rows)
    comparison_df = comparison_df.sort_values(by=["F1-Score", "ROC-AUC"], ascending=False).reset_index(drop=True)

    return comparison_df, eval_results


def plot_confusion_matrices(eval_results: dict, top_n: int = 4, plot_dir: str = "plots"):
    """
    Plots confusion matrix heatmaps for the top N models.
    """
    ensure_plot_dir(plot_dir)
    model_names = list(eval_results.keys())[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        cm = eval_results[name]["Confusion_Matrix"]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Healthy (0)', 'Cirrhosis (1)'],
                    yticklabels=['Healthy (0)', 'Cirrhosis (1)'])
        axes[idx].set_title(f"Confusion Matrix: {name}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Predicted Class")
        axes[idx].set_ylabel("Actual Class")

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "06_confusion_matrices.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved confusion matrices plot to {output_path}")


def plot_roc_curves(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series, plot_dir: str = "plots"):
    """
    Plots ROC Curves for all evaluated models on a single figure.
    """
    ensure_plot_dir(plot_dir)
    plt.figure(figsize=(10, 7))

    for name, pipeline in pipelines.items():
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_prob = pipeline.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "07_roc_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved ROC curves plot to {output_path}")


def plot_feature_importance(
    best_pipeline: Any,
    feature_names: list,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    best_model_name: str = "Best Model",
    plot_dir: str = "plots"
) -> pd.DataFrame:
    """
    Extracts and visualizes feature importances for the best model.
    Uses native tree/coef importances if available, or permutation importance as fallback.
    """
    ensure_plot_dir(plot_dir)
    classifier = best_pipeline.named_steps['classifier']

    importances = None
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        imp_type = "Native Gini / Split Importance"
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
        imp_type = "Absolute Logistic Coefficient Importance"
    elif X_test is not None and y_test is not None:
        perm_imp = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_imp.importances_mean
        imp_type = "Permutation Importance"
    else:
        raise ValueError("Could not extract feature importances for model.")

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='Blues_r')
    plt.title(f"Feature Importance ({best_model_name} - {imp_type})", fontsize=13, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.ylabel("Clinical Biomarker")

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "08_feature_importance.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved feature importance plot to {output_path}")

    return feat_imp_df
