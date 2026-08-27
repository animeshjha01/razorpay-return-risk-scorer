import unittest
import os
import sys
import tempfile
import sqlite3
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from api import app, lifespan
from pipeline_utils import MODEL_VERSION
import api

class TestAPI(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # We need the real model and policy for integration tests
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'return_risk_model.joblib'))
        self.policy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'policy_config.json'))
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.policy_path):
            raise unittest.SkipTest("Real model or policy artifacts missing; skipping API integration tests.")

        # Re-initialize the scoring service manually to inject a temporary audit log
        from scoring_service import ScoringService
        from audit_log import AuditLog
        
        self.fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.fd)
        
        self.audit_log = AuditLog(db_path=self.temp_db_path)
        
        api.scoring_service = ScoringService(
            model_path=self.model_path,
            policy_path=self.policy_path,
            audit_log=self.audit_log
        )
        api.service_initialization_error = None
        
        # TestClient creates a sync testing interface
        self.client = TestClient(app)
        
        self.base_order = {
            "order_id": "ord_api_123",
            "amount_inr": 1500.0,
            "method": "upi",
            "category": "home",
            "is_new_customer": 0,
            "past_orders": 5,
            "past_return_rate": 0.05,
            "order_hour": 14,
            "is_weekend": 0,
            "is_late_night": 0,
            "delivery_distance_km": 10.0,
            "checkout_time_sec": 45.0
        }

    async def asyncTearDown(self):
        try:
            if os.path.exists(self.temp_db_path):
                os.remove(self.temp_db_path)
        except PermissionError:
            pass

    # --- HEALTH TESTS ---
    def test_health_ready(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "READY")
        self.assertEqual(data["model_version"], MODEL_VERSION)
        self.assertEqual(data["policy_version"], "1.1")

    def test_health_not_ready(self):
        api.scoring_service = None
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "NOT_READY")

    # --- HAPPY PATH / INTEGRATION ---
    def test_score_order_success(self):
        response = self.client.post("/score-order", json=self.base_order)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["order_id"], "ord_api_123")
        self.assertIn("risk_score", data)
        self.assertIn("decision", data)
        self.assertIn("reason_codes", data)
        
        # Check audit consistency
        audit_record = self.audit_log.get_audit_record(data["audit_id"])
        self.assertIsNotNone(audit_record)
        self.assertEqual(audit_record["decision"], data["decision"])
        self.assertAlmostEqual(audit_record["risk_score"], data["risk_score"], places=5)
        self.assertEqual(audit_record["model_version"], data["model_version"])
        
    def test_score_order_unknown_category(self):
        order = self.base_order.copy()
        order["method"] = "bitcoin"
        response = self.client.post("/score-order", json=order)
        self.assertEqual(response.status_code, 200)

    # --- VALIDATION (400/422) ---
    def test_score_order_missing_field(self):
        order = self.base_order.copy()
        del order["amount_inr"]
        response = self.client.post("/score-order", json=order)
        self.assertEqual(response.status_code, 422) # FastAPI's Pydantic validation

    def test_score_order_invalid_type(self):
        order = self.base_order.copy()
        order["amount_inr"] = "not_a_number"
        response = self.client.post("/score-order", json=order)
        self.assertEqual(response.status_code, 422)

    def test_score_order_nan_rejection(self):
        order = self.base_order.copy()
        order["amount_inr"] = "NaN"
        response = self.client.post("/score-order", json=order)
        self.assertEqual(response.status_code, 422)
        
        # Verify no audit record was created
        with sqlite3.connect(self.temp_db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM decision_audit_log").fetchone()[0]
            self.assertEqual(count, 0)

    def test_score_order_infinity_rejection(self):
        order = self.base_order.copy()
        order["amount_inr"] = "Infinity"
        response = self.client.post("/score-order", json=order)
        self.assertEqual(response.status_code, 422)
        
        # Verify no audit record was created
        with sqlite3.connect(self.temp_db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM decision_audit_log").fetchone()[0]
            self.assertEqual(count, 0)

    def test_score_order_domain_validation_failure(self):
        order = self.base_order.copy()
        order["amount_inr"] = 49  # Below min
        response = self.client.post("/score-order", json=order)
        self.assertEqual(response.status_code, 400)
        self.assertIn("below the minimum", response.json()["detail"])
        
    def test_score_order_domain_digital_cod(self):
        order = self.base_order.copy()
        order["method"] = "cod"
        order["category"] = "digital"
        order["delivery_distance_km"] = 0
        response = self.client.post("/score-order", json=order)
        self.assertEqual(response.status_code, 400)

    # --- SERVICE FAILURES (503) ---
    def test_score_order_model_load_error(self):
        from unittest.mock import patch
        from scoring_service import ModelLoadError
        
        with patch('api.scoring_service.score_order', side_effect=ModelLoadError("Mocked model load error")):
            response = self.client.post("/score-order", json=self.base_order)
            self.assertEqual(response.status_code, 503)
            self.assertIn("unavailable", response.json()["detail"])

    def test_score_order_policy_load_error(self):
        from unittest.mock import patch
        from scoring_service import PolicyLoadError
        
        with patch('api.scoring_service.score_order', side_effect=PolicyLoadError("Mocked policy load error")):
            response = self.client.post("/score-order", json=self.base_order)
            self.assertEqual(response.status_code, 503)
            self.assertIn("unavailable", response.json()["detail"])

    def test_score_order_audit_failure(self):
        import stat
        os.chmod(self.temp_db_path, stat.S_IREAD)
        
        response = self.client.post("/score-order", json=self.base_order)
        self.assertEqual(response.status_code, 503)
        self.assertIn("recorded", response.json()["detail"])
        
        os.chmod(self.temp_db_path, stat.S_IWRITE | stat.S_IREAD)

if __name__ == '__main__':
    unittest.main()
