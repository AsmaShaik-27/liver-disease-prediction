"""
Data Loading, Preprocessing, and Cleaning Module for Liver Cirrhosis Prediction ML System.
Handles dataset loading, deduplication, missing value imputation, target encoding,
categorical encoding, and leakage-free train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_raw_data(file_path: str = "indian_liver_patient.csv") -> pd.DataFrame:
    """
    Loads raw dataset from CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Returns basic dataset summary statistics and schema information.
    """
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum())
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses raw liver patient DataFrame:
    1. Removes duplicate records to prevent data leakage/bias.
    2. Imputes missing values in 'Albumin_and_Globulin_Ratio' with column median.
    3. Encodes categorical column 'Gender' (Male=1, Female=0).
    4. Target Column Definition & Remapping:
       - Column name: 'Dataset'
       - Original values: 1 = Liver Patient (Disease Present), 2 = Non-Liver Patient (Healthy Control)
       - Remapped values: 1 -> 1 (Positive / Cirrhosis), 2 -> 0 (Negative / Healthy)
    """
    df_clean = df.copy()

    # 1. Remove duplicate records
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[Preprocessing] Removed {dup_count} duplicate rows. Clean dataset shape: {df_clean.shape}")

    # 2. Impute missing values in Albumin_and_Globulin_Ratio
    missing_ratio = df_clean['Albumin_and_Globulin_Ratio'].isnull().sum()
    if missing_ratio > 0:
        median_val = df_clean['Albumin_and_Globulin_Ratio'].median()
        df_clean['Albumin_and_Globulin_Ratio'] = df_clean['Albumin_and_Globulin_Ratio'].fillna(median_val)
        print(f"[Preprocessing] Imputed {missing_ratio} missing values in 'Albumin_and_Globulin_Ratio' with median: {median_val:.3f}")

    # 3. Categorical encoding for Gender: Male = 1, Female = 0
    if 'Gender' in df_clean.columns and df_clean['Gender'].dtype == 'object':
        df_clean['Gender'] = df_clean['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        print("[Preprocessing] Encoded 'Gender': Female -> 0, Male -> 1")

    # 4. Remap target column 'Dataset' to standard binary labels (1=Disease, 0=Healthy)
    if 'Dataset' in df_clean.columns:
        # Original: 1 -> Liver Patient, 2 -> Non-liver Patient
        df_clean['Dataset'] = df_clean['Dataset'].map({1: 1, 2: 0})
        print("[Preprocessing] Remapped target 'Dataset': 1 (Disease) -> 1, 2 (Healthy Control) -> 0")

    return df_clean


def split_data(df: pd.DataFrame, target_col: str = 'Dataset', test_size: float = 0.2, random_state: int = 42):
    """
    Splits processed DataFrame into feature matrix (X) and target vector (y),
    followed by a stratified 80:20 train-test split to preserve class ratio.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[Split] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[Split] Train class balance (1/0): {dict(pd.Series(y_train).value_counts())}")
    print(f"[Split] Test class balance (1/0): {dict(pd.Series(y_test).value_counts())}")

    return X_train, X_test, y_train, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fits StandardScaler strictly on training data to prevent data leakage.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_data(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """
    Transforms feature matrix X using a fitted StandardScaler.
    """
    return scaler.transform(X)
