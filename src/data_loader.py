import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

def get_snowflake_connection():
    """Establishes and returns a connection to Snowflake."""
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE")
    )

def load_data(query="SELECT * FROM CUSTOMER_ANALYTICS.ML.CHURN_FEATURES"):
    """Loads data from Snowflake using the provided query."""
    conn = get_snowflake_connection()
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        try:
            df = cursor.fetch_pandas_all()
        except Exception:
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()
    return df

