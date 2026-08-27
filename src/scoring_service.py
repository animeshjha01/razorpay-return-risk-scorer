import os
import json
import joblib
import pandas as pd
import math

from pipeline_utils import MODEL_VERSION
from decision_policy import process_decision
from audit_log import AuditLog, AuditLogError

class ScoringServiceError(Exception):
    """Base exception for scoring service."""
    pass

class ValidationError(ScoringServiceError):
    """Raised when incoming data fails structural schema validation."""
    pass

class DomainValidationError(ScoringServiceError):
    """Raised when data violates hard synthetic-domain rules."""
    pass

class ModelLoadError(ScoringServiceError):
    """Raised when the ML model cannot be loaded."""
    pass

class PolicyLoadError(ScoringServiceError):
    """Raised when the policy configuration cannot be loaded."""
    pass

class AuditFailureError(ScoringServiceError):
    """Raised when a decision cannot be durably written to the audit log."""
    pass

class ScoringService:
    def __init__(self, model_path: str = "models/return_risk_model.joblib",
                 policy_path: str = "models/policy_config.json",
                 audit_log: AuditLog = None):
        
        # 1. Load Model
        if not os.path.exists(model_path):
            raise ModelLoadError(f"Model file not found at {model_path}")
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {model_path}: {e}")

        # 2. Load Policy
        if not os.path.exists(policy_path):
            raise PolicyLoadError(f"Policy file not found at {policy_path}")
        try:
            with open(policy_path, 'r') as f:
                self.policy_config = json.load(f)
        except Exception as e:
            raise PolicyLoadError(f"Failed to load policy from {policy_path}: {e}")

        # 3. Setup Audit Log
        self.audit_log = audit_log if audit_log else AuditLog()

    def _validate_input(self, order: dict):
        """Validates the structure and types of the input order."""
        required_fields = [
            "order_id", "amount_inr", "method", "category", "is_new_customer",
            "past_orders", "past_return_rate", "order_hour", "is_weekend",
            "is_late_night", "delivery_distance_km", "checkout_time_sec"
        ]
        
        for field in required_fields:
            if field not in order:
                raise ValidationError(f"Missing required field: '{field}'")
                
        # Basic Type Checks
        if not order["order_id"] or not isinstance(order["order_id"], str):
            raise ValidationError("order_id must be a non-empty string.")
            
        if not isinstance(order["method"], str):
            raise ValidationError("method must be a string.")
            
        if not isinstance(order["category"], str):
            raise ValidationError("category must be a string.")
            
        numeric_fields = [
            "amount_inr", "past_orders", "past_return_rate", 
            "order_hour", "delivery_distance_km", "checkout_time_sec"
        ]
        
        for nf in numeric_fields:
            val = order[nf]
            if not isinstance(val, (int, float)):
                raise ValidationError(f"{nf} must be numeric.")
            if math.isnan(val) or math.isinf(val):
                raise ValidationError(f"{nf} must be a finite number.")
                
        # Binary Checks
        for bf in ["is_new_customer", "is_weekend", "is_late_night"]:
            val = order[bf]
            if val not in (0, 1, False, True):
                raise ValidationError(f"{bf} must be binary (0 or 1).")

    def _validate_domain_rules(self, order: dict):
        """Validates synthetic domain rules established by the generator."""
        amt = order["amount_inr"]
        method = str(order["method"]).lower()
        category = str(order["category"]).lower()
        dist = order["delivery_distance_km"]
        
        if amt < 50:
            raise DomainValidationError(f"Amount {amt} is below the minimum allowed (50 INR).")
            
        if method == "cod" and amt > 15000:
            raise DomainValidationError(f"COD amount {amt} exceeds maximum allowed for COD (15000 INR).")
            
        if category == "digital" and method == "cod":
            raise DomainValidationError("Digital goods cannot be purchased using COD.")
            
        if category == "digital" and dist != 0:
            raise DomainValidationError("Digital goods must have zero delivery distance.")
            
        if dist < 0:
            raise DomainValidationError("Delivery distance cannot be negative.")

    def score_order(self, order: dict) -> dict:
        """
        Orchestrates validation, prediction, decision, explanation, and auditing.
        Returns the structured result.
        """
        # 1. Validation
        self._validate_input(order)
        self._validate_domain_rules(order)
        
        # 2. Prediction
        try:
            df = pd.DataFrame([order])
            # Extracts probability for the positive class (risk)
            risk_score = float(self.model.predict_proba(df)[0, 1])
        except Exception as e:
            raise ScoringServiceError(f"Model prediction failed: {e}")
            
        # 3. Decision & 4. Explanation
        try:
            decision_result = process_decision(risk_score, order, self.policy_config, self.model)
        except Exception as e:
            raise ScoringServiceError(f"Decision processing failed: {e}")
            
        # 5. Audit Logging
        try:
            audit_id = self.audit_log.append_decision(
                order_id=order["order_id"],
                risk_score=decision_result["risk_score"],
                decision=decision_result["decision"],
                reason_codes=decision_result["reason_codes"],
                model_version=MODEL_VERSION,
                policy_version=decision_result["policy_version"]
            )
        except AuditLogError as e:
            # Crucially, we raise this explicitly so we do NOT return a misleading success.
            raise AuditFailureError(f"Failed to persist audit log: {e}")
            
        # 6. Return Structured Result
        decision_result["order_id"] = order["order_id"]
        decision_result["audit_id"] = audit_id
        decision_result["model_version"] = MODEL_VERSION
        
        return decision_result
