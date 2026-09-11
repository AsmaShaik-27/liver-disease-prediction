import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=random_state, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=random_state),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', random_state=random_state, n_jobs=1),
        "LightGBM": LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=random_state, force_col_wise=True, verbose=-1, n_jobs=1),
        "Support Vector Machine": SVC(probability=True, kernel='rbf', class_weight='balanced', random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    }
    return models


def create_pipeline(model: Any, use_smote: bool = True, random_state: int = 42) -> ImbPipeline:

    steps = [('scaler', StandardScaler())]
    if use_smote:
        steps.append(('smote', SMOTE(random_state=random_state)))
    steps.append(('classifier', model))
    
    pipeline = ImbPipeline(steps)
    return pipeline


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    use_smote: bool = True
) -> Dict[str, ImbPipeline]:

    models = get_candidate_models(random_state=random_state)
    fitted_pipelines = {}

    print(f"\n[Training] Starting candidate model training (SMOTE={use_smote})...")
    for model_name, model_inst in models.items():
        pipeline = create_pipeline(model_inst, use_smote=use_smote, random_state=random_state)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[model_name] = pipeline
        print(f"  -> Trained {model_name} successfully.")

    return fitted_pipelines


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, float]]:

    print("\n[Hyperparameter Tuning] Performing Stratified 5-Fold Grid Search on top models...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    param_grids = {
        "Random Forest (Tuned)": {
            "model": RandomForestClassifier(random_state=random_state, n_jobs=1),
            "grid": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [6, 8]
            }
        },
        "XGBoost (Tuned)": {
            "model": XGBClassifier(eval_metric='logloss', random_state=random_state, n_jobs=1),
            "grid": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [3, 4]
            }
        },
        "Gradient Boosting (Tuned)": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "grid": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [3, 4]
            }
        },
        "LightGBM (Tuned)": {
            "model": LGBMClassifier(random_state=random_state, force_col_wise=True, verbose=-1, n_jobs=1),
            "grid": {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [3, 4]
            }
        }
    }

    tuned_pipelines = {}
    best_cv_scores = {}

    for name, config in param_grids.items():
        pipeline = create_pipeline(config["model"], use_smote=True, random_state=random_state)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["grid"],
            cv=cv,
            scoring='roc_auc',
            n_jobs=1,
            error_score='raise'
        )
        grid_search.fit(X_train, y_train)

        tuned_pipelines[name] = grid_search.best_estimator_
        best_cv_scores[name] = grid_search.best_score_
        print(f"  -> {name} tuned! Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"     Best params: {grid_search.best_params_}")

    return tuned_pipelines, best_cv_scores
