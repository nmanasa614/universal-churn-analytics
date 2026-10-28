"""Train and evaluate churn models on the generated customer dataset."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "manufacturing_customers.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "tenure_months", "orders_last_90_days", "avg_order_value",
    "days_since_last_order", "late_delivery_rate", "complaint_count",
    "payment_delay_days", "discount_dependency", "order_frequency_change"
]
CATEGORICAL_FEATURES = ["region", "product_category", "customer_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset missing. Run: python generate_data.py")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "test_rows": int(len(y_test)),
        "churn_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Retained", "Churned"], zero_division=0
        )
    }

    joblib.dump(pipeline, MODEL_DIR / "churn_pipeline.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Test-set evaluation ===")
    for key in ["test_rows", "churn_rate_test", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        value = metrics[key]
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print("\n", metrics["classification_report"])
    print(f"Saved model to {MODEL_DIR / 'churn_pipeline.joblib'}")
    print("Use recall and precision together; do not judge the model by accuracy alone.")

if __name__ == "__main__":
    main()
