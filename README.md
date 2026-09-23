# Customer Churn Prediction System

An end-to-end machine learning project demonstrating data engineering with Snowflake, machine learning with Scikit-learn, and a business dashboard using Streamlit.

## Architecture
1.  **Snowflake Data Warehouse:** RAW, STAGING, and MART schemas containing the dimensional data models and feature engineering views.
2.  **ML Pipeline (`src/`):** A modular Python pipeline to connect to Snowflake, preprocess data, and train Logistic Regression and Random Forest models with threshold tuning.
3.  **Model Persistence (`models/`):** The final Logistic Regression model is saved via `joblib` alongside its threshold metadata for inference.
4.  **Dashboard (`streamlit_app.py`):** An interactive Streamlit dashboard allowing business users to filter and identify high-risk churn customers.
5.  **CI/CD:** Automated testing using `pytest` and GitHub Actions.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your Snowflake credentials. Never commit
`.env`; it is excluded by `.gitignore`.

Train the production model and write it to `models/`:

```powershell
python -m src.train
```

Run the Snowflake batch prediction pipeline:

```powershell
python -m src.predict
```

Run predictions for a local CSV file:

```powershell
python predict_from_csv.py --input .\new_customers.csv --output .\my_predictions.csv
```

Launch the dashboard:

```powershell
streamlit run streamlit_app.py
```

## Input data

`new_customers.csv` is a small example input file for local inference. The
`archive/` dataset is the public IBM Telco Customer Churn sample dataset.
Replace these files with approved, non-sensitive data only.

## Generated files

Trained model binaries, model metadata, prediction outputs, Python virtual
environments, test caches, and local environment files are intentionally
ignored. A fresh clone must run the training command before local inference.

## Testing

```powershell
pytest tests/
```
