import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_raw_data(file_path: str = "indian_liver_patient.csv") -> pd.DataFrame:

    df = pd.read_csv(file_path)
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum())
    }


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df_clean = df.copy()

    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[Preprocessing] Removed {dup_count} duplicate rows. Clean dataset shape: {df_clean.shape}")

    missing_ratio = df_clean['Albumin_and_Globulin_Ratio'].isnull().sum()
    if missing_ratio > 0:
        median_val = df_clean['Albumin_and_Globulin_Ratio'].median()
        df_clean['Albumin_and_Globulin_Ratio'] = df_clean['Albumin_and_Globulin_Ratio'].fillna(median_val)
        print(f"[Preprocessing] Imputed {missing_ratio} missing values in 'Albumin_and_Globulin_Ratio' with median: {median_val:.3f}")

    if 'Gender' in df_clean.columns and df_clean['Gender'].dtype == 'object':
        df_clean['Gender'] = df_clean['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        print("[Preprocessing] Encoded 'Gender': Female -> 0, Male -> 1")

    if 'Dataset' in df_clean.columns:
        df_clean['Dataset'] = df_clean['Dataset'].map({1: 1, 2: 0})
        print("[Preprocessing] Remapped target 'Dataset': 1 (Disease) -> 1, 2 (Healthy Control) -> 0")

    return df_clean


def split_data(df: pd.DataFrame, target_col: str = 'Dataset', test_size: float = 0.2, random_state: int = 42):

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

    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_data(scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:

    return scaler.transform(X)
