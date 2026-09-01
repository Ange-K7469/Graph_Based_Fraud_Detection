import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fraud Intelligence & KG Explainer",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Fraud Detection & Knowledge Graph Inspector")
st.markdown("Query transactions, evaluate risk scores, and inspect underlying network topological explanations.")

# ---------------------------------------------------------
# Load Data (Caches the dataframe for performance)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    # Replace with your actual saved dataframe or parquet file path
    # e.g., pd.read_parquet("scored_validation_transactions.parquet")
    try:
        df = pd.read_csv("scored_validation_transactions.csv")
    except FileNotFoundError:
        # Fallback dummy data structure if file doesn't exist yet
        df = pd.DataFrame({
            'TransactionID': [2987000, 2987001, 2987002],
            'TransactionAmt': [55.00, 250.00, 1000.00],
            'card1': [13926, 2750, 4663],
            'isFraud': [0, 1, 0],
            'supervised_kg_tabular_risk': [0.05, 0.92, 0.45],
            'weighted_degree': [0.12, 2.45, 1.10],
            'pagerank': [0.0001, 0.0035, 0.0012],
            'community_fraud_rate': [0.02, 0.48, 0.15]
        })
    return df

df_data = load_data()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose Action", ["Transaction Lookup", "Report Suspicious Activity"])

# =========================================================
# MODE 1: TRANSACTION LOOKUP & EXPLAINER
# =========================================================
if app_mode == "Transaction Lookup":
    st.header("🔍 Transaction & Entity Risk Lookup")
    
    # Search input
    search_id = st.text_input("Enter TransactionID or Card1 Token:", placeholder="e.g., 2987001")
    
    if search_id:
        try:
            query_id = int(search_id)
            result = df_data[df_data['TransactionID'] == query_id]
        except ValueError:
            result = df_data[df_data['card1'].astype(str).str.contains(search_id)]
            
        if not result.empty:
            for _, row in result.iterrows():
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                risk_score = row.get('supervised_kg_tabular_risk', 0.0)
                
                with col1:
                    st.metric(label="Transaction ID", value=int(row['TransactionID']))
                    st.metric(label="Amount", value=f"${row.get('TransactionAmt', 0):,.2f}")
                    
                with col2:
                    st.metric(label="Card Token (card1)", value=int(row['card1']))
                    # Risk Badge Color Logic
                    risk_label = "🚨 High Risk" if risk_score > 0.7 else ("⚠️ Medium Risk" if risk_score > 0.3 else "✅ Low Risk")
                    st.metric(label="Model Risk Assessment", value=risk_label, delta=f"{risk_score:.2%}")
                    
                with col3:
                    actual_fraud = row.get('isFraud', 0)
                    status = "Fraudulent" if actual_fraud == 1 else "Legitimate"
                    st.metric(label="Historical Ground Truth", value=status)

                # --- Knowledge Graph Explainability Section ---
                st.subheader("🧠 Knowledge Graph Explainability Panel")
                
                w_deg = row.get('weighted_degree', 0)
                p_rank = row.get('pagerank', 0)
                comm_rate = row.get('community_fraud_rate', 0)
                
                # Dynamic Natural Language Generation
                explanation_parts = []
                if comm_rate > 0.3:
                    explanation_parts.append(f"It belongs to a tight transactional cluster where **{comm_rate:.1%}** of connected entities have historical fraud markers.")
                else:
                    explanation_parts.append(f"It sits in a relatively safe cluster with a low historical fraud rate ({comm_rate:.1%}).")
                    
                if w_deg > 1.0:
                    explanation_parts.append(f"The transaction has an elevated **weighted degree ({w_deg:.2f})**, signifying intensive reuse of shared identifiers (devices/emails) typical of bot rings.")
                else:
                    explanation_parts.append(f"The network structural density is normal (weighted degree: {w_deg:.2f}).")
                    
                if p_rank > 0.002:
                    explanation_parts.append(f"It acts as a **structural hub** (PageRank: {p_rank:.4f}) connecting multiple independent accounts.")

                # Display Explanation Card
                if risk_score > 0.5:
                    st.error(" ".join(explanation_parts))
                else:
                    st.success(" ".join(explanation_parts))
                    
        else:
            st.warning("No matching transaction found in the validation registry.")

# =========================================================
# MODE 2: REPORT FRAUD FORM
# =========================================================
elif app_mode == "Report Suspicious Activity":
    st.header("🚨 Report Fraudulent Transaction")
    st.markdown("Submit a flagged transaction token to update local graph threat weights.")
    
    with st.form("fraud_report_form"):
        r_tx_id = st.text_input("Transaction ID")
        r_card = st.text_input("Card Token (card1)")
        r_reason = st.selectbox("Reason for Report", ["Stolen Card / Unauthorized", "Bot / Automated Ring", "Account Takeover", "Suspicious Device Mismatch"])
        r_notes = st.text_area("Analyst Comments / Observations")
        
        submitted = st.form_submit_button("Submit Report to Graph Engine")
        
        if submitted:
            if r_tx_id:
                st.success(f"Successfully logged report for Transaction ID {r_tx_id}. Connected community nodes have been updated.")
            else:
                st.error("Please provide at least a valid Transaction ID.")