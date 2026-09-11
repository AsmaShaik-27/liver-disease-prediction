#  Indian Liver Disease Prediction

An end-to-end **Machine Learning + Flask Web Application** that predicts whether a patient is likely to belong to the **liver disease positive or negative class** based on clinical biomarkers.

The project uses the **Indian Liver Patient Dataset (ILPD)** and a **LightGBM classification pipeline** with SMOTE for handling class imbalance.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace professional medical advice.

---

##  Features

* Machine Learning based liver disease prediction
* LightGBM classification model
* SMOTE for handling class imbalance
* Leakage-aware preprocessing pipeline
* 5-Fold Stratified Cross-Validation
* Comparison of 12 ML model configurations
* Probability-based risk estimation
* Low / Moderate / High risk categories
* Flask web application
* REST API with `/predict` endpoint
* Interactive and responsive frontend
* Quick patient profile presets
* Medical terminology tooltips
* Input validation and error handling
* Pre-trained model saved using Joblib

---

##  Machine Learning

### Dataset

The project uses the **Indian Liver Patient Dataset (ILPD)** from the UCI Machine Learning Repository.

* Original records: **583**
* Duplicate records removed: **13**
* Final records: **570**
* Features: **10**
* Target: **Liver Disease / No Liver Disease**

### Input Features

1. Age
2. Gender
3. Total Bilirubin
4. Direct Bilirubin
5. Alkaline Phosphatase
6. ALT
7. AST
8. Total Proteins
9. Albumin
10. Albumin/Globulin Ratio

### Preprocessing

The pipeline includes:

```text
Data Cleaning
     ↓
Deduplication
     ↓
Missing Value Imputation
     ↓
Gender Encoding
     ↓
Train/Test Split
     ↓
StandardScaler
     ↓
SMOTE
     ↓
LightGBM
```

SMOTE is applied only during model training to avoid data leakage into the test set.

---

##  Model Performance

The project evaluated **12 different model configurations**, including Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, SVM, and KNN.

### Best Model: LightGBM

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **77.19%** |
| Precision | **84.81%** |
| Recall    | **82.72%** |
| F1-Score  | **83.75%** |
| ROC-AUC   | **0.7875** |

LightGBM was selected based on its overall performance, particularly its F1-score and balance between precision and recall.

---

##  Web Application

The project includes a Flask-based web interface where users can enter patient information and receive:

* Prediction
* Positive-class probability
* Risk category

### Risk Categories

| Probability | Risk     |
| ----------- | -------- |
| `< 30%`     | Low      |
| `30% – 70%` | Moderate |
| `> 70%`     | High     |

The application also provides sample healthy and high-risk patient profiles for quick testing.

---

##  REST API

The application provides a JSON API through:

```text
POST /predict
```

Example request:

```json
{
  "age": 62,
  "gender": "Male",
  "total_bilirubin": 10.9,
  "direct_bilirubin": 5.5,
  "alkaline_phosphatase": 699,
  "alt": 64,
  "ast": 100,
  "total_proteins": 7.5,
  "albumin": 3.2,
  "ag_ratio": 0.74
}
```

Example response:

```json
{
  "prediction": "Liver Disease Positive",
  "probability": 96.94,
  "risk_category": "High"
}
```

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **LightGBM**
* **XGBoost**
* **imbalanced-learn / SMOTE**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Flask**
* **HTML5**
* **CSS3**
* **JavaScript**

---

##  Project Structure

```text
Indian-Liver-Disease-Prediction/
│
├── app.py
├── main.py
├── indian_liver_patient.csv
├── requirements.txt
├── README.md
│
├── models/
│   └── prediction_pipeline.joblib
│
├── plots/
│   ├── 01_target_distribution.png
│   ├── 02_numerical_distributions.png
│   ├── 03_categorical_distribution.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_feature_boxplots_by_target.png
│   ├── 06_confusion_matrices.png
│   ├── 07_roc_curves.png
│   └── 08_feature_importance.png
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── predict.py
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── main.js
```

---

##  How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Indian-Liver-Disease-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

### 4. Open in browser

```text
http://127.0.0.1:5000
```

---

##  Model Explainability

Feature importance from the LightGBM model identified the following as the most frequently used features:

1. **Age**
2. **Alkaline Phosphatase**
3. **Direct Bilirubin**
4. **Total Proteins**
5. **Albumin/Globulin Ratio**

Feature importance represents predictive association and should not be interpreted as biological causation.

---

## Limitations

* Relatively small dataset with **570 cleaned records**
* Dataset originates from a single geographical region
* Model predicts the dataset's **binary liver disease classification**, not a confirmed diagnosis of cirrhosis or a specific liver pathology
* External clinical validation has not been performed
* The system is intended for educational and research purposes

---

##  Future Improvements

* Add SHAP-based individual prediction explanations
* Evaluate on larger multi-center datasets
* Add Docker deployment
* Improve model calibration and external validation
* Extend the application with additional clinical features

---

## License

This project is intended for **educational and research purposes**.

---

## Author

**Shaik Asma**

If you found this project useful, consider giving the repository a ⭐.
