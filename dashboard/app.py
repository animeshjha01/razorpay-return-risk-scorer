import streamlit as st
import requests
import json
import os
import pandas as pd
import altair as alt

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="Return Risk Console", layout="wide")

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
# Header
st.title("Return Risk Console")
st.markdown("##### ML-powered return-risk decisioning")

# Health Check Header Strip
health = check_api_health()
api_status = "Online" if health.get("status") == "READY" else "Offline"
model_version = health.get("model_version", "Unknown")
policy_version = health.get("policy_version", "Unknown")

status_dot = "🟢" if api_status == "Online" else "🔴"
st.markdown(f"**{status_dot} API**: {api_status} &nbsp;&nbsp;|&nbsp;&nbsp; **🧠 Model**: {model_version} &nbsp;&nbsp;|&nbsp;&nbsp; **📜 Policy**: {policy_version}")
if api_status != "Online":
    st.warning("API is currently unavailable. Ensure the backend FastAPI service is running.")

st.divider()

# Load Policy Thresholds for visualization
policy = load_json_artifact("policy_config.json")
review_thresh = policy.get('review_threshold', 0.23) if policy else 0.23
hold_thresh = policy.get('hold_threshold', 0.64) if policy else 0.64

# Tabs
tab1, tab2, tab3 = st.tabs(["Live Scoring Console", "Audit Ledger", "System & Policy Analytics"])

with tab1:
    col_input, col_result = st.columns([1.0, 1.8])
    
    with col_input:
        with st.container(border=True):
            with st.form("scoring_form"):
                st.markdown("#### ORDER DETAILS")
                order_id = st.text_input("Order ID", value="demo_001")
                amount_inr = st.number_input("Amount (INR)", min_value=0.0, value=2500.0, step=100.0)
                method = st.selectbox("Payment Method", ["cod", "upi", "card", "netbanking", "wallet"])
                category = st.selectbox("Category", ["apparel", "electronics", "home", "beauty", "digital", "grocery"])
                
                st.markdown("#### CUSTOMER PROFILE")
                c1, c2 = st.columns(2)
                with c1:
                    is_new_customer = st.checkbox("New Customer")
                with c2:
                    past_orders = st.number_input("Past Orders", min_value=0, value=0, step=1)
                past_return_rate = st.slider("Past Return Rate", 0.0, 1.0, 0.0)
                
                st.markdown("#### FULFILLMENT / BEHAVIOR")
                f1, f2 = st.columns(2)
                with f1:
                    is_weekend = st.checkbox("Weekend Order")
                    is_late_night = st.checkbox("Late Night Order")
                with f2:
                    order_hour = st.slider("Order Hour (0-23)", 0, 23, 14)
                    
                delivery_distance_km = st.number_input("Delivery Distance (km)", min_value=0.0, value=15.0)
                checkout_time_sec = st.number_input("Checkout Time (sec)", min_value=0.0, value=60.0)
                
                submitted = st.form_submit_button("Evaluate Risk", type="primary", use_container_width=True)

    with col_result:
        with st.container(border=True):
            st.markdown("### Risk Assessment")
            if not submitted:
                st.info("Enter an order and evaluate its return risk.")
            else:
                if api_status != "Online":
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
                        
                        # Score and Decision
                        d_col1, d_col2 = st.columns(2)
                        decision = data.get('decision', 'UNKNOWN')
                        with d_col1:
                            if decision == "APPROVE":
                                st.success(f"## {decision}")
                            elif decision == "REVIEW":
                                st.warning(f"## {decision}")
                            else:
                                st.error(f"## {decision}")
                        with d_col2:
                            try:
                                score_fmt = f"{float(data.get('risk_score', 0)):.3f}"
                            except Exception:
                                score_fmt = str(data.get('risk_score', 'N/A'))
                            st.metric("Risk Score", score_fmt, help="Reasonably calibrated probability estimate with limitations.")
                            
                        # Threshold Visualization
                        st.markdown("**Risk Score Gauge:**")
                        try:
                            zones = pd.DataFrame([
                                {"zone": "APPROVE", "start": 0.0, "end": review_thresh, "color": "#d4edda"},
                                {"zone": "REVIEW", "start": review_thresh, "end": hold_thresh, "color": "#fff3cd"},
                                {"zone": "HOLD", "start": hold_thresh, "end": 1.0, "color": "#f8d7da"},
                            ])
                            score_val = float(data.get('risk_score', 0.0))
                            score_df = pd.DataFrame([{"score": score_val}])
                            
                            c_zones = alt.Chart(zones).mark_rect(opacity=0.7).encode(
                                x=alt.X('start:Q', scale=alt.Scale(domain=[0.0, 1.0]), axis=alt.Axis(title='Risk Score Range', values=[0.0, review_thresh, hold_thresh, 1.0])),
                                x2='end:Q',
                                color=alt.Color('color:N', scale=None, legend=None),
                                tooltip=['zone', 'start', 'end']
                            )
                            c_score = alt.Chart(score_df).mark_tick(color='black', thickness=4, size=40).encode(
                                x='score:Q',
                                tooltip=['score']
                            )
                            gauge = alt.layer(c_zones, c_score).properties(height=50)
                            st.altair_chart(gauge, use_container_width=True)
                        except Exception:
                            st.markdown(f"`0 ───────── {review_thresh:.2f} (REVIEW) ───────── {hold_thresh:.2f} (HOLD) ───────── 1.00`")
                        
                        st.success("Decision recorded successfully")
                        st.markdown(f"**Audit ID**: `{data.get('audit_id')}` &nbsp;&nbsp;|&nbsp;&nbsp; **Order ID**: `{data.get('order_id')}` &nbsp;&nbsp;|&nbsp;&nbsp; **Model Version**: `{data.get('model_version')}` &nbsp;&nbsp;|&nbsp;&nbsp; **Policy Version**: `{data.get('policy_version')}`")
                        
                        st.markdown("---")
                        
                        # ML Contribution Tornado Chart
                        st.markdown("#### Model Contribution (log-odds)")
                        tornado_data = []
                        for p in data.get("pos_contributions", []):
                            tornado_data.append({"feature": p.get("feature", ""), "value": float(p.get("contribution", 0.0)), "type": "Risk Increasing"})
                        for n in data.get("neg_contributions", []):
                            tornado_data.append({"feature": n.get("feature", ""), "value": float(n.get("contribution", 0.0)), "type": "Risk Reducing"})
                            
                        if tornado_data:
                            try:
                                df_torn = pd.DataFrame(tornado_data)
                                df_torn['abs_value'] = df_torn['value'].abs()
                                df_torn = df_torn.sort_values('abs_value', ascending=False)
                                
                                tornado_chart = alt.Chart(df_torn).mark_bar().encode(
                                    x=alt.X('value:Q', title='Model Contribution (log-odds)'),
                                    y=alt.Y('feature:N', sort=alt.EncodingSortField(field="abs_value", order="descending"), title=''),
                                    color=alt.Color('type:N', scale=alt.Scale(domain=['Risk Increasing', 'Risk Reducing'], range=['#d9534f', '#5cb85c']), legend=alt.Legend(title="Factor Type")),
                                    tooltip=['feature', 'value']
                                ).properties(height=max(150, len(df_torn)*30))
                                st.altair_chart(tornado_chart, use_container_width=True)
                            except Exception:
                                st.info("Could not render model contribution chart.")
                        else:
                            st.info("No model contribution details available for this decision.")
                        
                        st.markdown("---")
                        
                        # Explanation (Detailed Factors)
                        e_col1, e_col2 = st.columns(2)
                        with e_col1:
                            with st.container(border=True):
                                st.markdown("#### Risk Increasing Factors")
                                pos = data.get("pos_contributions", [])
                                if pos:
                                    for p in pos:
                                        st.markdown(f"- **{p['feature']}** ({p['raw_value']}): *Model contribution +{p['contribution']:.2f}*")
                                else:
                                    st.markdown("*None*")
                        with e_col2:
                            with st.container(border=True):
                                st.markdown("#### Risk Reducing Factors")
                                neg = data.get("neg_contributions", [])
                                if neg:
                                    for n in neg:
                                        st.markdown(f"- **{n['feature']}** ({n['raw_value']}): *Model contribution {n['contribution']:.2f}*")
                                else:
                                    st.markdown("*None*")
                                
                        st.markdown("---")
                        
                        col_reason1, col_reason2 = st.columns(2)
                        with col_reason1:
                            with st.container(border=True):
                                st.markdown("#### Score Reason")
                                st.info(data.get('score_reason', ''))
                        with col_reason2:
                            with st.container(border=True):
                                reasons = data.get("reason_codes", [])
                                st.markdown("#### Reason Codes")
                                if reasons:
                                    for rc in reasons:
                                        st.markdown(f"- `{rc}`")
                                else:
                                    st.markdown("*None*")
                                
                        domain_sigs = data.get("domain_signals", [])
                        if domain_sigs:
                            with st.container(border=True):
                                st.markdown("#### Domain Signals")
                                for ds in domain_sigs:
                                    st.markdown(f"- `{ds}`")

                    elif resp.status_code == 400:
                        st.error(f"Domain Validation Error: {resp.json().get('detail', 'Invalid order constraints')}")
                    elif resp.status_code == 422:
                        st.error("Malformed Request: The order contains invalid data types or missing fields.")
                    else:
                        st.error(f"Service Unavailable: {resp.json().get('detail', 'Backend error.')}")


with tab2:
    st.markdown("### Audit Ledger")
    
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
        cols = ["timestamp", "order_id", "risk_score", "decision", "audit_id", "model_version", "policy_version"]
        df_display = df_audits[[c for c in cols if c in df_audits.columns]].copy()
        
        df_display.rename(columns={
            "audit_id": "Audit ID",
            "timestamp": "Time",
            "order_id": "Order ID",
            "risk_score": "Risk Score",
            "decision": "Decision",
            "reason_codes": "Reasons",
            "model_version": "Model Version",
            "policy_version": "Policy Version"
        }, inplace=True)
        
        def color_decision(val):
            if val == 'APPROVE':
                return 'color: #155724; background-color: #d4edda; font-weight: bold;'
            elif val == 'REVIEW':
                return 'color: #856404; background-color: #fff3cd; font-weight: bold;'
            elif val == 'HOLD':
                return 'color: #721c24; background-color: #f8d7da; font-weight: bold;'
            return ''
            
        try:
            styled_df = df_display.style.map(color_decision, subset=['Decision']).format({"Risk Score": "{:.3f}"})
        except AttributeError:
            # Fallback for older pandas versions
            styled_df = df_display.style.applymap(color_decision, subset=['Decision']).format({"Risk Score": "{:.3f}"})
            
        st.dataframe(styled_df, hide_index=True, use_container_width=True)
        
        st.markdown("#### Audit Receipt")
        selected_audit_id = st.selectbox("Select an Audit ID to view receipt:", [""] + list(df_audits["audit_id"]))
        if selected_audit_id:
            try:
                resp = requests.get(f"{API_BASE_URL}/audit/{selected_audit_id}", timeout=3)
                if resp.status_code == 200:
                    audit_data = resp.json()
                    with st.container(border=True):
                        st.markdown("#### AUDIT RECEIPT")
                        
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            st.markdown("##### Core Decision")
                            st.markdown(f"**Audit ID**: `{audit_data.get('audit_id')}`")
                            st.markdown(f"**Time**: `{audit_data.get('timestamp')}`")
                            st.markdown(f"**Order ID**: `{audit_data.get('order_id')}`")
                            st.markdown(f"**Decision**: `{audit_data.get('decision')}`")
                            try:
                                st.markdown(f"**Risk Score**: `{float(audit_data.get('risk_score', 0)):.3f}`")
                            except Exception:
                                st.markdown(f"**Risk Score**: `{audit_data.get('risk_score')}`")
                            st.markdown(f"**Model Version**: `{audit_data.get('model_version')}`")
                            st.markdown(f"**Policy Version**: `{audit_data.get('policy_version')}`")
                        with r_col2:
                            st.markdown("##### Decision Explanation")
                            if "score_reason" in audit_data:
                                st.markdown(f"**Score Reason**: {audit_data.get('score_reason')}")
                            reasons = audit_data.get("reason_codes", [])
                            if reasons:
                                st.markdown("**Reason Codes**:")
                                for r in reasons:
                                    st.markdown(f"- `{r}`")
                            pos = audit_data.get("pos_contributions", [])
                            if pos:
                                st.markdown("**Risk Increasing Factors**:")
                                for p in pos:
                                    st.markdown(f"- **{p.get('feature', '')}** ({p.get('raw_value', '')}): *+{p.get('contribution', '')}*")
                            neg = audit_data.get("neg_contributions", [])
                            if neg:
                                st.markdown("**Risk Reducing Factors**:")
                                for n in neg:
                                    st.markdown(f"- **{n.get('feature', '')}** ({n.get('raw_value', '')}): *{n.get('contribution', '')}*")
                            domain = audit_data.get("domain_signals", [])
                            if domain:
                                st.markdown("**Domain Signals**:")
                                for ds in domain:
                                    st.markdown(f"- `{ds}`")
                                    
                        with st.expander("View Raw Audit Record"):
                            st.json(audit_data)
                            
                elif resp.status_code == 404:
                    st.warning("Audit record not found.")
                else:
                    st.error("Failed to load audit record.")
            except Exception:
                st.error("Error communicating with the API.")


with tab3:
    st.markdown("### System & Policy Analytics")
    st.info("Model/policy analytics. Not live operational KPIs.")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("#### Final Held-Out Test Performance")
        metrics = load_json_artifact("test_metrics.json")
        if metrics:
            m1, m2 = st.columns(2)
            m1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            m2.metric("Average Precision", f"{metrics.get('average_precision', 0):.3f}")
            m1.metric("Brier Score", f"{metrics.get('brier_score', 0):.3f}")
            m2.metric("Positive-Event Prevalence", f"{metrics.get('prevalence', 0):.3f}")
        else:
            st.info("Held-out metrics unavailable — evaluation artifact not present in this environment.")
            
        st.markdown("#### Business Policy")
        if policy:
            p1, p2 = st.columns(2)
            p1.metric("Review Threshold", f"{policy.get('review_threshold', 0):.3f}")
            p2.metric("Hold Threshold", f"{policy.get('hold_threshold', 0):.3f}")
            p1.metric("Max Intervention Rate", f"{policy.get('max_intervention_rate', 0):.1%}")
            p2.metric("Max Hold Rate", f"{policy.get('max_hold_rate', 0):.1%}")
        else:
            st.info("Policy configuration unavailable — artifact not present in this environment.")
            
    with m_col2:
        st.markdown("#### Threshold Diagnostics")
        diagnostics = load_csv_artifact("test_diagnostics.csv")
        if diagnostics is not None and not diagnostics.empty:
            st.dataframe(diagnostics[['threshold', 'precision', 'recall', 'f1_score', 'flagged_rate']], hide_index=True)
        else:
            st.info("Threshold diagnostics unavailable — evaluation artifact not present in this environment.")
            
    st.divider()
    
    st.markdown("#### Operational Trade-off Visualization")
    candidates = load_csv_artifact("policy_candidates.csv")
    if candidates is not None and not candidates.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Chart A: Intervention Capacity vs Risk Recall**")
            chart_a = alt.Chart(candidates).mark_line(point=True).encode(
                x=alt.X('total_intervention_rate:Q', title='Intervention Capacity (Rate)'),
                y=alt.Y('risky_order_recall:Q', title='Risk Recall'),
                tooltip=['total_intervention_rate', 'risky_order_recall', 'review_threshold']
            ).interactive()
            st.altair_chart(chart_a, use_container_width=True)
        with c2:
            st.markdown("**Chart B: Intervention Capacity vs Estimated Cost**")
            chart_b = alt.Chart(candidates).mark_line(point=True, color='red').encode(
                x=alt.X('total_intervention_rate:Q', title='Intervention Capacity (Rate)'),
                y=alt.Y('total_estimated_cost:Q', title='Estimated Total Cost (INR)'),
                tooltip=['total_intervention_rate', 'total_estimated_cost', 'review_threshold']
            ).interactive()
            st.altair_chart(chart_b, use_container_width=True)
    else:
        st.info("Operational trade-off visualizations unavailable — candidates artifact not present in this environment.")
