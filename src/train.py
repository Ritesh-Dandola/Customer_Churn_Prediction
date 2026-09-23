import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data
from src.preprocessing import get_preprocessor
from src.evaluate import evaluate_model

def train():
    print("Loading data from Snowflake...")
    df = load_data()
    
    X = df.drop(columns=["CHURN", "CUSTOMER_ID"])
    y = df["CHURN"]
    
    print("Splitting data into 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("\nCreating internal validation set for tuning...")
    X_train_model, X_validation, y_train_model, y_validation = train_test_split(
        X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
    )
    
    preprocessor = get_preprocessor()
    
    # Logistic Regression Tuning
    logistic_val_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    logistic_val_pipeline.fit(X_train_model, y_train_model)
    log_val_prob = logistic_val_pipeline.predict_proba(X_validation)[:, 1]
    
    thresholds = np.arange(0.20, 0.71, 0.05)
    best_log_threshold = 0.5
    best_log_f1 = -1
    from sklearn.metrics import f1_score
    for t in thresholds:
        pred = np.where(log_val_prob >= t, "Yes", "No")
        f1 = f1_score(y_validation, pred, pos_label="Yes", zero_division=0)
        if f1 > best_log_f1:
            best_log_f1 = f1
            best_log_threshold = t
            
    print(f"Best Logistic Regression validation threshold: {best_log_threshold:.2f}")

    # Random Forest Tuning
    rf_val_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])
    rf_val_pipeline.fit(X_train_model, y_train_model)
    rf_val_prob = rf_val_pipeline.predict_proba(X_validation)[:, 1]
    
    best_rf_threshold = 0.5
    best_rf_f1 = -1
    for t in thresholds:
        pred = np.where(rf_val_prob >= t, "Yes", "No")
        f1 = f1_score(y_validation, pred, pos_label="Yes", zero_division=0)
        if f1 > best_rf_f1:
            best_rf_f1 = f1
            best_rf_threshold = t
            
    print(f"Best Random Forest validation threshold: {best_rf_threshold:.2f}")
    
    # Train Final Production Model (Logistic Regression as baseline per spec)
    # Spec requests to train final production model on ALL data, but for evaluation we first train on train split
    
    print("\nTraining Logistic Regression on full training data (for test set evaluation)...")
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    final_pipeline.fit(X_train, y_train)
    
    test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    # Using the exact threshold from specification: 0.40
    metrics = evaluate_model(y_test, test_prob, threshold=0.40)
    print("\nFinal Logistic Regression Test Metrics (Threshold=0.40):")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\nTraining Production Model on ALL data (7043 rows)...")
    production_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    production_pipeline.fit(X, y)

    # Save model and metadata
    os.makedirs("models", exist_ok=True)
    model_path = "models/logistic_churn_v1.joblib"
    joblib.dump(production_pipeline, model_path)
    print(f"\nProduction Model saved to {model_path}")
    
    metadata = {
        "model_name": "Logistic Regression",
        "model_version": "logistic_v1.0",
        "threshold": 0.40,
        "evaluation_metrics": metrics
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Metadata saved to models/model_metadata.json")

if __name__ == "__main__":
    train()

