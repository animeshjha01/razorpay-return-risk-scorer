import math
import json
from reason_codes import generate_reasons

def load_policy_config(config_path: str) -> dict:
    """Loads the frozen policy configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_score(score: float) -> float:
    """
    Validates that the risk score is a finite numeric float between 0 and 1.
    """
    try:
        score = float(score)
    except (TypeError, ValueError):
        raise ValueError(f"Risk score must be a numeric value, got: {type(score).__name__}")
        
    if math.isnan(score) or math.isinf(score):
        raise ValueError(f"Risk score must be finite, got: {score}")
        
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"Risk score must be between 0.0 and 1.0 inclusive, got: {score}")
        
    return score

def validate_policy(policy_config: dict) -> tuple[float, float]:
    """
    Validates the policy configuration and extracts the thresholds.
    """
    if 'review_threshold' not in policy_config or 'hold_threshold' not in policy_config:
        raise ValueError("Policy configuration is missing required threshold fields.")
        
    t_rev = float(policy_config['review_threshold'])
    t_hold = float(policy_config['hold_threshold'])
    
    if not (0.0 <= t_rev <= 1.0):
        raise ValueError(f"Review threshold must be between 0.0 and 1.0, got: {t_rev}")
        
    if not (0.0 <= t_hold <= 1.0):
        raise ValueError(f"Hold threshold must be between 0.0 and 1.0, got: {t_hold}")
        
    if t_rev >= t_hold:
        raise ValueError(f"Review threshold ({t_rev}) must be strictly less than hold threshold ({t_hold}).")
        
    return t_rev, t_hold

def decide(score: float, policy_config: dict) -> str:
    """
    Pure deterministic function to apply the frozen policy decision rule.
    """
    score = validate_score(score)
    t_rev, t_hold = validate_policy(policy_config)
    
    if score < t_rev:
        return "APPROVE"
    elif score < t_hold:
        return "REVIEW"
    else:
        return "HOLD"

def process_decision(score: float, features: dict, policy_config: dict, model=None) -> dict:
    """
    Combines the decision logic and reason-code generation into a structured output.
    This does NOT perform any external IO (no DB, no APIs).
    """
    decision = decide(score, policy_config)
    reasons = generate_reasons(features, validate_score(score), policy_config, model)
    
    # Collect flat reason codes for quick summary
    flat_reason_codes = [reasons["risk_score_reason"].split(":")[0]]
    for c in reasons["top_positive_model_contributions"]:
        flat_reason_codes.append(c["reason_code"])
    for c in reasons["top_negative_model_contributions"]:
        flat_reason_codes.append(c["reason_code"])
    flat_reason_codes.extend(reasons["domain_signals"])
    
    return {
        "decision": decision,
        "risk_score": float(score),
        "top_positive_model_contributions": reasons["top_positive_model_contributions"],
        "top_negative_model_contributions": reasons["top_negative_model_contributions"],
        "reason_codes": flat_reason_codes,
        "domain_signals": reasons["domain_signals"],
        "risk_score_reason": reasons["risk_score_reason"],
        "review_threshold": float(policy_config['review_threshold']),
        "hold_threshold": float(policy_config['hold_threshold']),
        "policy_version": policy_config.get('policy_version', 'UNKNOWN')
    }
