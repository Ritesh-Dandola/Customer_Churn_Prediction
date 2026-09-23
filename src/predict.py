import json
import joblib
import numpy as np
import pandas as pd
from datetime import date

def assign_risk_level(probability):
    if probability < 0.30:
        return 'LOW'
    elif probability < 0.60:
        return 'MEDIUM'
    else:
        return 'HIGH'

def predict_all(df):
    """
    Loads the trained model and predicts churn for all customers in the provided DataFrame.
    Output conforms to the ML.CHURN_PREDICTIONS schema.
    """
    try:
        model = joblib.load("models/logistic_churn_v1.joblib")
        with open("models/model_metadata.json", "r") as f:
            metadata = json.load(f)
            threshold = metadata.get("threshold", 0.40)
            model_version = metadata.get("model_version", "logistic_v1.0")
    except FileNotFoundError:
        raise FileNotFoundError("Model or metadata not found. Run src/train.py first.")
        
    X_new = df.drop(columns=["CHURN", "CUSTOMER_ID"], errors='ignore')
    
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = np.where(probabilities >= threshold, "Yes", "No")
    
    output_df = pd.DataFrame({
        "CUSTOMER_ID": df["CUSTOMER_ID"],
        "CHURN_PROBABILITY": probabilities,
        "PREDICTED_CHURN": predictions,
        "RISK_LEVEL": [assign_risk_level(p) for p in probabilities],
        "MODEL_VERSION": model_version,
        "PREDICTION_DATE": date.today().isoformat()
    })
    
    return output_df

if __name__ == "__main__":
    from src.data_loader import load_data
    from src.snowflake_writer import write_predictions_to_snowflake
    
    print("Loading data for batch prediction...")
    df = load_data()
    
    print("Running predictions...")
    predictions_df = predict_all(df)
    
    print("Writing predictions to Snowflake...")
    write_predictions_to_snowflake(predictions_df)

