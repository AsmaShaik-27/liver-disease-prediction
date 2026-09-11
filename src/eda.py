

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})


def ensure_plot_dir(plot_dir: str = "plots"):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


def plot_target_distribution(df: pd.DataFrame, plot_dir: str = "plots"):

    ensure_plot_dir(plot_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    target_counts = df['Dataset'].value_counts()
    labels = ['Liver Patient (1)', 'Non-Liver Patient (0)']
    colors = ['#d9534f', '#5bc0de']

    # Subplot 1: Bar chart
    sns.barplot(x=target_counts.index, y=target_counts.values, ax=axes[0], palette=colors, hue=target_counts.index, legend=False)
    axes[0].set_title("Target Class Counts (Cirrhosis Outcome)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Dataset Class (1 = Disease, 0 = Healthy)")
    axes[0].set_ylabel("Patient Count")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', fontsize=11, color='white', fontweight='bold')

    # Subplot 2: Pie chart
    axes[1].pie(target_counts.values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
                explode=(0.05, 0), textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1].set_title("Target Class Proportion", fontsize=13, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "01_target_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved target distribution plot to {output_path}")


def plot_numerical_distributions(df: pd.DataFrame, plot_dir: str = "plots"):

    ensure_plot_dir(plot_dir)
    num_cols = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
                'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for idx, col in enumerate(num_cols):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[idx], color='#337ab7', bins=25)
            axes[idx].set_title(f"Distribution of {col}", fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "02_numerical_distributions.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved numerical distributions plot to {output_path}")


def plot_categorical_distribution(df: pd.DataFrame, raw_df: pd.DataFrame, plot_dir: str = "plots"):

    ensure_plot_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    gender_col = raw_df['Gender'] if 'Gender' in raw_df.columns else df['Gender']
    sns.countplot(data=df, x=gender_col, hue='Dataset', palette=['#5bc0de', '#d9534f'], ax=ax)

    ax.set_title("Cirrhosis Outcome Breakdown by Gender", fontsize=13, fontweight='bold')
    ax.set_xlabel("Gender")
    ax.set_ylabel("Patient Count")
    ax.legend(['Non-Liver Patient (0)', 'Liver Patient (1)'], title="Outcome")

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "03_categorical_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved categorical distribution plot to {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, plot_dir: str = "plots"):

    ensure_plot_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(11, 9))

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, linewidths=.5, ax=ax, cbar_kws={"shrink": .8})

    ax.set_title("Pearson Correlation Heatmap of Clinical Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(plot_dir, "04_correlation_heatmap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved correlation heatmap plot to {output_path}")


def plot_feature_boxplots(df: pd.DataFrame, plot_dir: str = "plots"):

    ensure_plot_dir(plot_dir)
    key_features = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Albumin']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(key_features):
        if feature in df.columns:
            sns.boxplot(data=df, x='Dataset', y=feature, ax=axes[idx], palette=['#5bc0de', '#d9534f'], hue='Dataset', legend=False)
            axes[idx].set_title(f"{feature} by Outcome Class", fontsize=11, fontweight='bold')
            axes[idx].set_xlabel("Dataset (0=Healthy, 1=Liver Patient)")
            axes[idx].set_ylabel(feature)
            # Log scale for highly skewed biomarkers to improve visualization readability
            if df[feature].skew() > 2:
                axes[idx].set_yscale('log')
                axes[idx].set_ylabel(f"{feature} (Log Scale)")

    plt.tight_layout()
    output_path = os.path.join(plot_dir, "05_feature_boxplots_by_target.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved feature boxplots plot to {output_path}")


def run_eda(df: pd.DataFrame, raw_df: pd.DataFrame = None, plot_dir: str = "plots") -> dict:

    if raw_df is None:
        raw_df = df

    plot_target_distribution(df, plot_dir)
    plot_numerical_distributions(df, plot_dir)
    plot_categorical_distribution(df, raw_df, plot_dir)
    plot_correlation_heatmap(df, plot_dir)
    plot_feature_boxplots(df, plot_dir)

    # Compute correlation with target
    target_corr = df.corr()['Dataset'].drop('Dataset').sort_values(ascending=False)

    summary_findings = {
        "total_records": len(df),
        "target_breakdown": df['Dataset'].value_counts().to_dict(),
        "positive_class_percentage": float((df['Dataset'] == 1).mean() * 100),
        "top_positive_correlations": target_corr.head(3).to_dict(),
        "top_negative_correlations": target_corr.tail(3).to_dict(),
        "highly_correlated_pairs": [
            ("Total_Bilirubin", "Direct_Bilirubin", float(df['Total_Bilirubin'].corr(df['Direct_Bilirubin']))),
            ("Alamine_Aminotransferase", "Aspartate_Aminotransferase", float(df['Alamine_Aminotransferase'].corr(df['Aspartate_Aminotransferase']))),
            ("Total_Protiens", "Albumin", float(df['Total_Protiens'].corr(df['Albumin']))),
            ("Albumin", "Albumin_and_Globulin_Ratio", float(df['Albumin'].corr(df['Albumin_and_Globulin_Ratio'])))
        ]
    }
    return summary_findings
