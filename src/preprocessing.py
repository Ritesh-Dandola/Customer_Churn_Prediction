from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def get_preprocessor():
    """Returns the scikit-learn ColumnTransformer for data preprocessing."""
    numeric_features = [
        "SENIOR_CITIZEN",
        "TENURE",
        "MONTHLY_CHARGES",
        "TOTAL_CHARGES",
        "NUMBER_OF_SERVICES",
        "HAS_ONLINE_SECURITY",
        "HAS_ONLINE_BACKUP",
        "HAS_DEVICE_PROTECTION",
        "HAS_TECH_SUPPORT",
        "IS_MONTH_TO_MONTH",
        "IS_ELECTRONIC_PAYMENT"
    ]

    categorical_features = [
        "GENDER",
        "PARTNER",
        "DEPENDENTS",
        "PHONE_SERVICE",
        "MULTIPLE_LINES",
        "INTERNET_SERVICE",
        "ONLINE_SECURITY",
        "ONLINE_BACKUP",
        "DEVICE_PROTECTION",
        "TECH_SUPPORT",
        "STREAMING_TV",
        "STREAMING_MOVIES",
        "CONTRACT",
        "PAPERLESS_BILLING",
        "PAYMENT_METHOD",
        "TENURE_GROUP"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )
    
    return preprocessor

