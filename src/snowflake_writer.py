from snowflake.connector.pandas_tools import write_pandas
from src.data_loader import get_snowflake_connection

def write_predictions_to_snowflake(df, table_name="CHURN_PREDICTIONS", schema="ML"):
    """
    Writes prediction DataFrame to Snowflake ML schema.
    """
    conn = get_snowflake_connection()
    try:
        # Switch to correct schema
        conn.cursor().execute(f"USE SCHEMA {schema}")
        
        # Write to snowflake
        success, nchunks, nrows, _ = write_pandas(conn, df, table_name.upper(), auto_create_table=True, overwrite=True)
        print(f"Successfully wrote {nrows} rows to {schema}.{table_name}")
        return success
    finally:
        conn.close()

