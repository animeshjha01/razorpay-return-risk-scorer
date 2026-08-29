import pandas as pd
import numpy as np

# Heuristic thresholds downgraded to domain-context only
# These do NOT masquerade as model explanations anymore.
DOMAIN_CONTEXT_THRESHOLDS = {
    "HIGH_ORDER_VALUE_INR": 10000.0,
    "LONG_DELIVERY_DISTANCE_KM": 50.0,
}

# The minimum absolute contribution in log-odds space required to surface 
# a feature in the user-facing explanation payload. This removes negligible 
# mathematical noise (e.g. checkout_time_sec = -0.0025) while preserving
# the actual underlying calculation and risk decision.
EXPLANATION_MIN_ABS_CONTRIBUTION = 0.05

# Mapping raw feature + direction to human-readable codes and text
MODEL_REASON_MAPPINGS = {
    "past_return_rate_increases_risk": ("ELEVATED_HISTORICAL_RETURN_RATE", "Higher historical return rate provides a risk-increasing model contribution in log-odds space."),
    "past_return_rate_reduces_risk": ("LOW_HISTORICAL_RETURN_RATE", "Low historical return rate provides a risk-reducing model contribution in log-odds space."),
    "method_increases_risk": ("ELEVATED_RISK_PAYMENT_METHOD", "This payment method provides a risk-increasing model contribution in log-odds space."),
    "method_reduces_risk": ("LOWER_RISK_PAYMENT_METHOD", "This payment method provides a risk-reducing model contribution in log-odds space."),
    "category_increases_risk": ("ELEVATED_RISK_CATEGORY", "This product category provides a risk-increasing model contribution in log-odds space."),
    "category_reduces_risk": ("LOWER_RISK_CATEGORY", "This product category provides a risk-reducing model contribution in log-odds space."),
    "is_new_customer_increases_risk": ("NEW_CUSTOMER_RISK", "New customer profile provides a risk-increasing model contribution in log-odds space."),
    "is_new_customer_reduces_risk": ("ESTABLISHED_CUSTOMER", "Established customer profile provides a risk-reducing model contribution in log-odds space."),
    "amount_inr_increases_risk": ("ELEVATED_ORDER_AMOUNT", "The order amount provides a risk-increasing model contribution in log-odds space."),
    "amount_inr_reduces_risk": ("LOWER_ORDER_AMOUNT", "The order amount provides a risk-reducing model contribution in log-odds space."),
    "delivery_distance_km_increases_risk": ("ELEVATED_DELIVERY_DISTANCE", "The delivery distance provides a risk-increasing model contribution in log-odds space."),
    "delivery_distance_km_reduces_risk": ("LOWER_DELIVERY_DISTANCE", "The delivery distance provides a risk-reducing model contribution in log-odds space."),
    "is_late_night_increases_risk": ("LATE_NIGHT_ORDER_RISK", "The time of order provides a risk-increasing model contribution in log-odds space."),
    "is_late_night_reduces_risk": ("DAYTIME_ORDER", "The time of order provides a risk-reducing model contribution in log-odds space."),
    "past_orders_increases_risk": ("FEW_PAST_ORDERS", "Low volume of past orders provides a risk-increasing model contribution in log-odds space."),
    "past_orders_reduces_risk": ("HIGH_PAST_ORDERS", "High volume of past orders provides a risk-reducing model contribution in log-odds space.")
}

def _get_reason_code_and_text(feature_name: str, direction: str, raw_value: any) -> tuple[str, str]:
    key = f"{feature_name}_{direction}"
    if key in MODEL_REASON_MAPPINGS:
        return MODEL_REASON_MAPPINGS[key]
    dir_str = "risk-increasing" if direction == "increases_risk" else "risk-reducing"
    return (
        f"MODEL_CONTRIBUTION_{feature_name.upper()}",
        f"The {feature_name} feature provides a {dir_str} model contribution in log-odds space."
    )

def extract_model_contributions(model, features_dict: dict) -> list[dict]:
    """
    Given a fitted Logistic Regression pipeline and a single feature dict,
    calculates actual model-consistent contributions for each raw feature.
    """
    from pipeline_utils import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, PASSTHROUGH_FEATURES
    
    record_df = pd.DataFrame([features_dict])
    
    # Transform
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    
    X_transformed = preprocessor.transform(record_df)
    if hasattr(X_transformed, 'toarray'):
        X_arr = X_transformed.toarray()[0]
    else:
        X_arr = X_transformed[0]
        
    coefs = classifier.coef_[0]
    raw_contributions = X_arr * coefs
    
    # Get feature names matching the transformed array
    feature_names = []
    feature_names.extend(NUMERIC_FEATURES)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_names)
    feature_names.extend(PASSTHROUGH_FEATURES)
    
    # Aggregate back to raw features
    agg_contributions = {f: 0.0 for f in FEATURES}
    
    for name, cont in zip(feature_names, raw_contributions):
        if name in FEATURES:
            agg_contributions[name] += cont
        else:
            for cat_feat in CATEGORICAL_FEATURES:
                if name.startswith(cat_feat + '_'):
                    agg_contributions[cat_feat] += cont
                    break

    explanations = []
    for raw_f in FEATURES:
        c = agg_contributions[raw_f]
        raw_val = features_dict.get(raw_f)
        
        # Filter out literal zeros and pure floating-point noise from OneHotEncoder
        # The true materiality filter for explanations is applied later.
        if abs(c) < 1e-6:
            continue
            
        direction = "increases_risk" if c > 0 else "reduces_risk"
        code, text = _get_reason_code_and_text(raw_f, direction, raw_val)
        
        explanations.append({
            "feature": raw_f,
            "raw_value": raw_val,
            "contribution": float(c),
            "direction": direction,
            "reason_code": code,
            "reason_text": text
        })
        
    # Sort by absolute magnitude of contribution descending
    return sorted(explanations, key=lambda x: abs(x['contribution']), reverse=True)

def generate_reasons(features: dict, risk_score: float, policy_config: dict, model=None) -> dict:
    """
    Generates a structured explanation payload combining model contributions,
    domain signals, and risk score categorization.
    """
    result = {
        "score_reason": "",
        "pos_contributions": [],
        "neg_contributions": [],
        "domain_signals": []
    }
    
    # 1. Risk Score categorizations
    t_rev = policy_config.get('review_threshold', 1.0)
    t_hold = policy_config.get('hold_threshold', 1.0)
    
    if risk_score < t_rev:
        result["score_reason"] = "LOW_MODEL_RISK: The reasonably calibrated probability estimate indicates low model risk."
    elif risk_score < t_hold:
        result["score_reason"] = "ELEVATED_MODEL_RISK: The reasonably calibrated probability estimate indicates elevated model risk."
    else:
        result["score_reason"] = "HIGH_MODEL_RISK: The reasonably calibrated probability estimate indicates high model risk."

    # 2. Model contributions
    if model is not None:
        contributions = extract_model_contributions(model, features)
        
        # Apply human-facing materiality threshold
        meaningful_contributions = [c for c in contributions if abs(c['contribution']) >= EXPLANATION_MIN_ABS_CONTRIBUTION]
        
        pos_contribs = [c for c in meaningful_contributions if c['direction'] == 'increases_risk']
        neg_contribs = [c for c in meaningful_contributions if c['direction'] == 'reduces_risk']
        
        result["pos_contributions"] = pos_contribs[:3]
        result["neg_contributions"] = neg_contribs[:3]
        
    # 3. Domain context (heuristics)
    if features.get('amount_inr', 0.0) >= DOMAIN_CONTEXT_THRESHOLDS["HIGH_ORDER_VALUE_INR"]:
        result["domain_signals"].append("DOMAIN_SIGNAL_HIGH_ORDER_VALUE")
        
    if features.get('delivery_distance_km', 0.0) >= DOMAIN_CONTEXT_THRESHOLDS["LONG_DELIVERY_DISTANCE_KM"]:
        result["domain_signals"].append("DOMAIN_SIGNAL_LONG_DELIVERY_DISTANCE")
        
    return result
