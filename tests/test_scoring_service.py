import unittest
import os
import tempfile
import sys
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from scoring_service import (
    ScoringService, ValidationError, DomainValidationError, 
    ModelLoadError, PolicyLoadError, AuditFailureError
)
from audit_log import AuditLog
from pipeline_utils import MODEL_VERSION

class TestScoringService(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # We need the real model and policy for integration tests
        cls.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'return_risk_model.joblib'))
        cls.policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'policy_config.json'))
        
        if not os.path.exists(cls.model_path) or not os.path.exists(cls.policy_path):
            raise unittest.SkipTest("Real model or policy artifacts missing; skipping scoring integration tests.")

    def setUp(self):
        # Setup temporary audit log
        self.fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.fd)
        
        self.audit_log = AuditLog(db_path=self.temp_db_path)
        
        # Initialize service
        self.service = ScoringService(
            model_path=self.model_path,
            policy_path=self.policy_path,
            audit_log=self.audit_log
        )
        
        self.base_order = {
            "order_id": "ord_123",
            "amount_inr": 1500,
            "method": "upi",
            "category": "home",
            "is_new_customer": 0,
            "past_orders": 5,
            "past_return_rate": 0.05,
            "order_hour": 14,
            "is_weekend": 0,
            "is_late_night": 0,
            "delivery_distance_km": 10,
            "checkout_time_sec": 45
        }

    def tearDown(self):
        try:
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
        except PermissionError:
            pass

    # --- HAPPY PATH & INTEGRATION CORRECTNESS ---
    
    def test_complete_end_to_end_flow(self):
        """Verifies integration across model, decision, explanation, and audit."""
        result = self.service.score_order(self.base_order)
        
        # Output schema validation
        self.assertEqual(result["order_id"], "ord_123")
        self.assertIn("risk_score", result)
        self.assertIn("decision", result)
        self.assertIn("reason_codes", result)
        self.assertIn("audit_id", result)
        self.assertEqual(result["model_version"], MODEL_VERSION)
        
        # Audit consistency
        audit_record = self.audit_log.get_audit_record(result["audit_id"])
        self.assertIsNotNone(audit_record)
        
        self.assertEqual(audit_record["order_id"], result["order_id"])
        self.assertEqual(audit_record["decision"], result["decision"])
        self.assertAlmostEqual(audit_record["risk_score"], result["risk_score"], places=6)
        self.assertEqual(audit_record["reason_codes"], result["reason_codes"])
        self.assertEqual(audit_record["model_version"], result["model_version"])
        self.assertEqual(audit_record["policy_version"], result["policy_version"])
        
        # Explanation check (At least some reasons exist)
        self.assertGreater(len(result["reason_codes"]), 0)

    # --- MODEL BEHAVIOR ---
    
    def test_unknown_categorical_values(self):
        """Unknown categories should not crash (OneHotEncoder ignore)."""
        order = self.base_order.copy()
        order["method"] = "bitcoin"
        order["category"] = "spaceship"
        
        # Should not raise exception
        result = self.service.score_order(order)
        self.assertIn("risk_score", result)
        self.assertIsNotNone(result["decision"])

    def test_missing_required_feature(self):
        order = self.base_order.copy()
        del order["amount_inr"]
        with self.assertRaises(ValidationError):
            self.service.score_order(order)
            
    def test_invalid_numeric_type(self):
        order = self.base_order.copy()
        order["past_orders"] = "five"
        with self.assertRaises(ValidationError):
            self.service.score_order(order)
            
    def test_nan_infinity_rejected(self):
        order = self.base_order.copy()
        order["amount_inr"] = float('inf')
        with self.assertRaises(ValidationError):
            self.service.score_order(order)

    # --- DOMAIN VALIDATION ---
    
    def test_domain_negative_amount(self):
        order = self.base_order.copy()
        order["amount_inr"] = -100
        with self.assertRaises(DomainValidationError):
            self.service.score_order(order)
            
    def test_domain_amount_below_minimum(self):
        order = self.base_order.copy()
        order["amount_inr"] = 49
        with self.assertRaises(DomainValidationError):
            self.service.score_order(order)
            
    def test_domain_digital_plus_cod(self):
        order = self.base_order.copy()
        order["category"] = "digital"
        order["method"] = "cod"
        order["delivery_distance_km"] = 0
        with self.assertRaises(DomainValidationError):
            self.service.score_order(order)
            
    def test_domain_digital_plus_distance(self):
        order = self.base_order.copy()
        order["category"] = "digital"
        order["method"] = "upi"
        order["delivery_distance_km"] = 10
        with self.assertRaises(DomainValidationError):
            self.service.score_order(order)
            
    def test_domain_cod_above_max(self):
        order = self.base_order.copy()
        order["method"] = "cod"
        order["amount_inr"] = 16000
        with self.assertRaises(DomainValidationError):
            self.service.score_order(order)

    # --- FAILURE HANDLING ---
    
    def test_missing_model_produces_load_failure(self):
        with self.assertRaises(ModelLoadError):
            ScoringService(model_path="invalid/path.joblib", policy_path=self.policy_path)
            
    def test_invalid_policy_produces_load_failure(self):
        with self.assertRaises(PolicyLoadError):
            ScoringService(model_path=self.model_path, policy_path="invalid/path.json")

    def test_audit_write_failure(self):
        # Make the database strictly read-only to trigger AuditFailureError
        import stat
        os.chmod(self.temp_db_path, stat.S_IREAD)
        
        with self.assertRaises(AuditFailureError):
            self.service.score_order(self.base_order)
            
        os.chmod(self.temp_db_path, stat.S_IWRITE | stat.S_IREAD)

if __name__ == '__main__':
    unittest.main()
