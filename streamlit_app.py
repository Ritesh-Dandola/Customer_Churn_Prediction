import streamlit as st
import pandas as pd
from src.data_loader import get_snowflake_connection

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")

st.title("Customer Churn Analytics")

@st.cache_data(ttl=600)
def load_predictions():
    conn = get_snowflake_connection()
    cursor = None
    query = """
        SELECT P.*, B.CONTRACT, B.PAYMENT_METHOD, S.INTERNET_SERVICE 
        FROM CUSTOMER_ANALYTICS.ML.CHURN_PREDICTIONS P
        LEFT JOIN CUSTOMER_ANALYTICS.MART.FACT_CUSTOMER_BILLING B ON P.CUSTOMER_ID = B.CUSTOMER_ID
        LEFT JOIN CUSTOMER_ANALYTICS.STAGING.TELCO_CUSTOMERS S ON P.CUSTOMER_ID = S.CUSTOMER_ID
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        try:
            df = cursor.fetch_pandas_all()
        except Exception:
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=columns)
    except Exception as e:
        # Fallback to local data if Snowflake table doesn't exist yet for demo purposes
        st.error(f"Error fetching from Snowflake: {e}")
        return pd.DataFrame()
    finally:
        if cursor is not None:
            cursor.close()
        conn.close()
    return df

df = load_predictions()

if df.empty:
    st.warning("No prediction data found. Please run the prediction pipeline first.")
    st.stop()

# KPIs
total_customers = len(df)
predicted_churners = len(df[df['PREDICTED_CHURN'] == 'Yes'])
high_risk = len(df[df['RISK_LEVEL'] == 'HIGH'])
avg_prob = df['CHURN_PROBABILITY'].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Predicted Churners", f"{predicted_churners:,}")
col3.metric("High Risk", f"{high_risk:,}")
col4.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

st.divider()

# Layout
c1, c2 = st.columns(2)

with c1:
    st.subheader("Risk Distribution")
    risk_dist = df['RISK_LEVEL'].value_counts().reset_index()
    risk_dist.columns = ['Risk Level', 'Count']
    st.bar_chart(risk_dist.set_index('Risk Level'))

with c2:
    st.subheader("High Risk Customers by Contract")
    high_risk_df = df[df['RISK_LEVEL'] == 'HIGH']
    contract_dist = high_risk_df['CONTRACT'].value_counts()
    st.bar_chart(contract_dist)

st.divider()

st.subheader("High Risk Customers")
# Search & Filter
search_id = st.text_input("Search Customer ID")
risk_filter = st.selectbox("Filter Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])

filtered_df = df.copy()
if risk_filter != "All":
    filtered_df = filtered_df[filtered_df['RISK_LEVEL'] == risk_filter]
if search_id:
    filtered_df = filtered_df[filtered_df['CUSTOMER_ID'].str.contains(search_id, case=False, na=False)]

st.dataframe(
    filtered_df[['CUSTOMER_ID', 'CHURN_PROBABILITY', 'RISK_LEVEL', 'CONTRACT', 'PREDICTED_CHURN', 'PREDICTION_DATE']],
    width="stretch"
)

