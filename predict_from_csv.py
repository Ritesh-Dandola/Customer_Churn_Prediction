"""
predict_from_csv.py
-------------------
Run predictions on new, unseen customer data from a CSV file.

Usage:
    python predict_from_csv.py --input path/to/new_customers.csv --output path/to/predictions.csv

The input CSV must contain the same columns as the original IBM Telco dataset.
The script will clean the data, run it through the saved model, and save
the predictions to a CSV file. Optionally, it can also write to Snowflake.
"""

import argparse
import pandas as pd
import numpy as np
import json
import joblib
from datetime import date

# ──────────────────────────────────────────────
# Risk level assignment (same rule as pipeline)
# ──────────────────────────────────────────────
def assign_risk_level(probability):
    if probability < 0.30:
        return 'LOW'
    elif probability < 0.60:
        return 'MEDIUM'
    else:
        return 'HIGH'


# ──────────────────────────────────────────────
# Clean raw CSV into the same feature format
# that the model was trained on
# ──────────────────────────────────────────────
def preprocess_raw_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a raw Telco CSV DataFrame (with original column names)
    into the feature format expected by the model (matching ML.CHURN_FEATURES).
    """
    df = df.copy()

    # Rename columns to uppercase to match Snowflake convention
    df.columns = [c.upper() for c in df.columns]

    # Rename specific columns to match training feature names
    rename_map = {
        "CUSTOMERID": "CUSTOMER_ID",
        "SENIORCITIZEN": "SENIOR_CITIZEN",
        "PHONESERVICE": "PHONE_SERVICE",
        "MULTIPLELINES": "MULTIPLE_LINES",
        "INTERNETSERVICE": "INTERNET_SERVICE",
        "ONLINESECURITY": "ONLINE_SECURITY",
        "ONLINEBACKUP": "ONLINE_BACKUP",
        "DEVICEPROTECTION": "DEVICE_PROTECTION",
        "TECHSUPPORT": "TECH_SUPPORT",
        "STREAMINGTV": "STREAMING_TV",
        "STREAMINGMOVIES": "STREAMING_MOVIES",
        "PAPERLESSBILLING": "PAPERLESS_BILLING",
        "PAYMENTMETHOD": "PAYMENT_METHOD",
        "MONTHLYCHARGES": "MONTHLY_CHARGES",
        "TOTALCHARGES": "TOTAL_CHARGES",
    }
    df.rename(columns=rename_map, inplace=True)

    # Cast numeric columns - TotalCharges can be a blank string
    df["TENURE"] = pd.to_numeric(df["TENURE"], errors="coerce")
    df["MONTHLY_CHARGES"] = pd.to_numeric(df["MONTHLY_CHARGES"], errors="coerce")
    df["TOTAL_CHARGES"] = pd.to_numeric(
        df["TOTAL_CHARGES"].astype(str).str.strip().replace("", np.nan),
        errors="coerce"
    )
    df["SENIOR_CITIZEN"] = pd.to_numeric(df["SENIOR_CITIZEN"], errors="coerce")

    # Feature engineering (mirrors sql/06_feature_engineering.sql)
    df["TENURE_GROUP"] = pd.cut(
        df["TENURE"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-1 Year", "1-2 Years", "2-4 Years", "4+ Years"]
    ).astype(str)

    df["HAS_ONLINE_SECURITY"] = (df["ONLINE_SECURITY"] == "Yes").astype(int)
    df["HAS_ONLINE_BACKUP"] = (df["ONLINE_BACKUP"] == "Yes").astype(int)
    df["HAS_DEVICE_PROTECTION"] = (df["DEVICE_PROTECTION"] == "Yes").astype(int)
    df["HAS_TECH_SUPPORT"] = (df["TECH_SUPPORT"] == "Yes").astype(int)
    df["IS_MONTH_TO_MONTH"] = (df["CONTRACT"] == "Month-to-month").astype(int)
    df["IS_ELECTRONIC_PAYMENT"] = df["PAYMENT_METHOD"].isin(
        ["Electronic check", "Credit card (automatic)"]
    ).astype(int)

    df["NUMBER_OF_SERVICES"] = (
        (df["PHONE_SERVICE"] == "Yes").astype(int) +
        (df["INTERNET_SERVICE"] != "No").astype(int) +
        df["HAS_ONLINE_SECURITY"] +
        df["HAS_ONLINE_BACKUP"] +
        df["HAS_DEVICE_PROTECTION"] +
        df["HAS_TECH_SUPPORT"] +
        (df["STREAMING_TV"] == "Yes").astype(int) +
        (df["STREAMING_MOVIES"] == "Yes").astype(int)
    )

    return df


# ──────────────────────────────────────────────
# Main prediction function
# ──────────────────────────────────────────────
def predict_from_csv(input_path: str, output_path: str, write_to_snowflake: bool = False):
    print(f"Loading new customer data from: {input_path}")
    raw_df = pd.read_csv(input_path)
    print(f"Loaded {len(raw_df)} records.")

    print("Applying feature engineering...")
    df = preprocess_raw_csv(raw_df)

    print("Loading model and metadata...")
    try:
        model = joblib.load("models/logistic_churn_v1.joblib")
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
            threshold = metadata.get("threshold", 0.40)
            model_version = metadata.get("model_version", "logistic_v1.0")
    except FileNotFoundError:
        raise FileNotFoundError("Model not found. Run 'python -m src.train' first.")

    # Extract only the feature columns the model was trained on
    drop_cols = ["CUSTOMER_ID", "CHURN"]
    X_new = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"Running predictions with threshold={threshold}...")
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = np.where(probabilities >= threshold, "Yes", "No")

    output_df = pd.DataFrame({
        "CUSTOMER_ID": df["CUSTOMER_ID"] if "CUSTOMER_ID" in df.columns else range(len(df)),
        "CHURN_PROBABILITY": probabilities.round(4),
        "PREDICTED_CHURN": predictions,
        "RISK_LEVEL": [assign_risk_level(p) for p in probabilities],
        "MODEL_VERSION": model_version,
        "PREDICTION_DATE": date.today().isoformat()
    })

    output_df.to_csv(output_path, index=False)
    print(f"\nDone! Predictions saved to: {output_path}")

    # Summary
    churners = (output_df["PREDICTED_CHURN"] == "Yes").sum()
    high_risk = (output_df["RISK_LEVEL"] == "HIGH").sum()
    print(f"\n--- Summary ---")
    print(f"Total customers   : {len(output_df)}")
    print(f"Predicted churners: {churners} ({churners/len(output_df)*100:.1f}%)")
    print(f"High risk         : {high_risk}")

    if write_to_snowflake:
        from src.snowflake_writer import write_predictions_to_snowflake
        print("\nWriting predictions to Snowflake ML.CHURN_PREDICTIONS...")
        write_predictions_to_snowflake(output_df)

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict churn from a new CSV file.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", default="predictions_output.csv", help="Path to save predictions CSV.")
    parser.add_argument("--snowflake", action="store_true", help="Also write predictions to Snowflake.")
    args = parser.parse_args()

    predict_from_csv(args.input, args.output, args.snowflake)
