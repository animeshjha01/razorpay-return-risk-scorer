import streamlit as st
import requests
import json
import os
import pandas as pd
import altair as alt

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="AI Return-Risk Scorer", layout="wide")

# --- Helper Functions ---
def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "NOT_READY", "error": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "NOT_READY", "error": "Connection failed"}

def score_order(payload: dict):
    resp = requests.post(f"{API_BASE_URL}/score-order", json=payload, timeout=5)
    return resp

def load_json_artifact(filename: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def load_csv_artifact(filename: str) -> pd.DataFrame:
    path = os.path.join(ARTIFACTS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# --- Main Layout ---
st.title("AI Return-Risk Scorer")
st.subheader("Defense-only risk decisioning for e-commerce orders")

st.markdown("""
**Demo environment — synthetic transaction data**  
*Disclaimer: This system is trained on synthetic data using hypothetical business-cost assumptions. It does not reflect real Razorpay production performance. It is a defense-only system intended to assist human review and does not autonomously block real payments.*
""")

# --- Health Check ---
health = check_api_health()
if health.get("status") == "READY":
    st.success(f"Backend API READY | Model: {health.get('model_version')} | Policy: {health.get('policy_version')}")
else:
    st.error(f"Risk API unavailable. Start the FastAPI service and try again. ({health.get('error')})")

st.divider()

# --- Live Risk Scoring Form ---
st.header("Live Risk Scoring")
with st.form("scoring_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        order_id = st.text_input("Order ID", value="demo_001")
        amount_inr = st.number_input("Amount (INR)", min_value=0.0, value=2500.0, step=100.0)
        method = st.selectbox("Payment Method", ["cod", "upi", "card", "netbanking", "wallet"])
        category = st.selectbox("Category", ["apparel", "electronics", "home", "beauty", "digital", "grocery"])
        
    with col2:
        is_new_customer = st.checkbox("New Customer")
        past_orders = st.number_input("Past Orders", min_value=0, value=0, step=1)
        past_return_rate = st.slider("Past Return Rate", 0.0, 1.0, 0.0)
        order_hour = st.slider("Order Hour (0-23)", 0, 23, 14)
        
    with col3:
        is_weekend = st.checkbox("Weekend Order")
        is_late_night = st.checkbox("Late Night Order")
        delivery_distance_km = st.number_input("Delivery Distance (km)", min_value=0.0, value=15.0)
        checkout_time_sec = st.number_input("Checkout Time (sec)", min_value=0.0, value=60.0)

    submitted = st.form_submit_button("Score Order", type="primary")

if submitted:
    if health.get("status") != "READY":
        st.error("Cannot score order. API is not ready.")
    else:
        payload = {
            "order_id": order_id,
            "amount_inr": amount_inr,
            "method": method,
            "category": category,
            "is_new_customer": 1 if is_new_customer else 0,
            "past_orders": past_orders,
            "past_return_rate": past_return_rate,
            "order_hour": order_hour,
            "is_weekend": 1 if is_weekend else 0,
            "is_late_night": 1 if is_late_night else 0,
            "delivery_distance_km": delivery_distance_km,
            "checkout_time_sec": checkout_time_sec
        }
        
        with st.spinner("Evaluating risk..."):
            resp = score_order(payload)
            
        if resp.status_code == 200:
            data = resp.json()
            
            st.subheader("Risk Result")
            st.info(f"**Decision audited successfully** | Audit ID: `{data['audit_id']}` | Order: `{data['order_id']}` | Model: `{data['model_version']}` | Policy: `{data['policy_version']}`")
            
            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                decision = data['decision']
                if decision == "APPROVE":
                    st.success(f"### Decision: {decision}")
                elif decision == "REVIEW":
                    st.warning(f"### Decision: {decision}")
                else:
                    st.error(f"### Decision: {decision}")
                    
                st.metric("Risk probability estimate", f"{data['risk_score']:.3f}", help="Reasonably calibrated probability estimate with limitations.")
                st.caption(f"Reason: {data.get('risk_score_reason', '')}")
            
            with r_col2:
                st.markdown("### Why did the model reach this decision?")
                
                pos_contribs = data.get("top_positive_model_contributions", [])
                neg_contribs = data.get("top_negative_model_contributions", [])
                
                if pos_contribs:
                    st.markdown("**Top risk-increasing model contributions**")
                    st.table(pd.DataFrame(pos_contribs)[['feature', 'raw_value', 'contribution', 'reason_text']])
                    
                if neg_contribs:
                    st.markdown("**Top risk-reducing model contributions**")
                    st.table(pd.DataFrame(neg_contribs)[['feature', 'raw_value', 'contribution', 'reason_text']])
            
            # Reason Codes
            st.markdown("### Reason Codes")
            st.markdown("**Model-derived reasons:**")
            for rc in data.get("reason_codes", []):
                if not rc.startswith("DOMAIN_SIGNAL"):
                    st.markdown(f"- `{rc}`")
                    
            domain_sigs = data.get("domain_signals", [])
            if domain_sigs:
                st.markdown("**Domain-context signals:**")
                for ds in domain_sigs:
                    st.markdown(f"- `{ds}`")
                    
        elif resp.status_code == 400:
            st.error(f"Domain Validation Error: {resp.json().get('detail', 'Invalid order constraints')}")
        elif resp.status_code == 422:
            st.error("Malformed Request: The order contains invalid data types or missing fields.")
        elif resp.status_code == 503:
            st.error(f"Service Unavailable: {resp.json().get('detail', 'Backend model, policy, or audit log unavailable.')}")
        else:
            st.error("Internal Server Error: An unexpected error occurred while communicating with the scoring service.")

st.divider()

# --- Model & Policy Sections ---
st.header("System Evaluation & Policy Assumptions")

m_col1, m_col2 = st.columns(2)

with m_col1:
    st.subheader("Final Held-Out Test Performance")
    metrics = load_json_artifact("test_metrics.json")
    if metrics:
        st.markdown("*Note: The test set was strictly held out from model fitting, policy selection, and threshold tuning.*")
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        m2.metric("Average Precision", f"{metrics.get('average_precision', 0):.3f}")
        m1.metric("Brier Score", f"{metrics.get('brier_score', 0):.3f}")
        m2.metric("Positive-Event Prevalence", f"{metrics.get('prevalence', 0):.3f}")
    else:
        st.warning("Metrics artifact not found or corrupted.")

    st.subheader("Business Policy")
    policy = load_json_artifact("policy_config.json")
    if policy:
        st.markdown("**Synthetic simulation assumptions** (Not Razorpay facts)")
        p1, p2 = st.columns(2)
        p1.metric("Review Threshold", f"{policy.get('review_threshold', 0):.3f}")
        p2.metric("Hold Threshold", f"{policy.get('hold_threshold', 0):.3f}")
        p1.metric("Max Intervention Rate", f"{policy.get('max_intervention_rate', 0):.1%}")
        p2.metric("Max Hold Rate", f"{policy.get('max_hold_rate', 0):.1%}")
        p1.metric("Review Cost Assumption", f"₹{policy.get('cost_review', 0)}")
        p2.metric("Hold Cost Assumption", f"₹{policy.get('cost_hold', 0)}")
    else:
        st.warning("Policy artifact not found or corrupted.")

with m_col2:
    st.subheader("FINAL HELD-OUT TEST DIAGNOSTICS — REPORTING ONLY")
    diagnostics = load_csv_artifact("test_diagnostics.csv")
    if diagnostics is not None and not diagnostics.empty:
        st.dataframe(diagnostics[['threshold', 'precision', 'recall', 'f1_score', 'flagged_rate']], hide_index=True)
    else:
        st.warning("Diagnostics artifact not found or corrupted.")

st.divider()

# --- Policy Operational Trade-offs ---
st.header("Operational Trade-off Visualization")
st.info("The operating policy is selected on an internal validation split using explicit cost and operational-capacity assumptions. The final model and frozen policy are then evaluated on an untouched test set. Changing business assumptions can change the selected policy.")

candidates = load_csv_artifact("policy_candidates.csv")
if candidates is not None and not candidates.empty:
    st.markdown("**Chart A: Intervention Capacity vs Risk Recall**")
    st.markdown("*Less review capacity → fewer risky orders captured → higher estimated loss*")
    
    # Chart A: Intervention Rate vs Recall
    chart_a = alt.Chart(candidates).mark_line(point=True).encode(
        x=alt.X('total_intervention_rate:Q', title='Intervention Capacity (Rate)'),
        y=alt.Y('risky_order_recall:Q', title='Risk Recall'),
        tooltip=['total_intervention_rate', 'risky_order_recall', 'review_threshold']
    ).interactive()
    st.altair_chart(chart_a, use_container_width=True)

    st.markdown("**Chart B: Intervention Capacity vs Estimated Cost**")
    # Chart B: Intervention Rate vs Cost
    chart_b = alt.Chart(candidates).mark_line(point=True, color='red').encode(
        x=alt.X('total_intervention_rate:Q', title='Intervention Capacity (Rate)'),
        y=alt.Y('total_estimated_cost:Q', title='Estimated Total Cost (INR)'),
        tooltip=['total_intervention_rate', 'total_estimated_cost', 'review_threshold']
    ).interactive()
    st.altair_chart(chart_b, use_container_width=True)
else:
    st.warning("Policy candidates artifact not found or corrupted.")

st.divider()
st.header("Recent Audited Decisions")

@st.cache_data(ttl=5)
def get_recent_audits():
    try:
        resp = requests.get(f"{API_BASE_URL}/audit/recent?limit=20", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []

recent_records = get_recent_audits()
if not recent_records:
    st.info("No recent audit records found or API is unavailable.")
else:
    df_audits = pd.DataFrame(recent_records)
    # Reorder/select useful columns
    cols = ["timestamp", "order_id", "risk_score", "decision", "audit_id", "model_version", "policy_version"]
    df_display = df_audits[[c for c in cols if c in df_audits.columns]]
    st.dataframe(df_display, hide_index=True, use_container_width=True)
    
    st.subheader("Inspect Audit Record")
    selected_audit_id = st.selectbox("Select an Audit ID to view full details:", [""] + list(df_audits["audit_id"]))
    if selected_audit_id:
        try:
            resp = requests.get(f"{API_BASE_URL}/audit/{selected_audit_id}", timeout=3)
            if resp.status_code == 200:
                st.json(resp.json())
            elif resp.status_code == 404:
                st.warning("Audit record not found.")
            else:
                st.error("Failed to load audit record.")
        except Exception:
            st.error("Error communicating with the API.")
