import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd

# Add dashboard to path for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard')))
import app as dashboard_app

class TestDashboardHelperFunctions(unittest.TestCase):
    
    @patch('dashboard.app.requests.get')
    def test_check_api_health_ready(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "READY", "model_version": "V1", "policy_version": "1.0"}
        mock_get.return_value = mock_resp
        
        health = dashboard_app.check_api_health()
        self.assertEqual(health["status"], "READY")
        self.assertEqual(health["model_version"], "V1")

    @patch('dashboard.app.requests.get')
    def test_check_api_health_not_ready(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_get.return_value = mock_resp
        
        health = dashboard_app.check_api_health()
        self.assertEqual(health["status"], "NOT_READY")

    @patch('dashboard.app.requests.post')
    def test_score_order_calls_api(self, mock_post):
        payload = {"order_id": "test"}
        dashboard_app.score_order(payload)
        mock_post.assert_called_once_with(f"{dashboard_app.API_BASE_URL}/score-order", json=payload, timeout=5)

    @patch('dashboard.app.os.path.exists')
    def test_load_json_artifact_missing(self, mock_exists):
        mock_exists.return_value = False
        res = dashboard_app.load_json_artifact("missing.json")
        self.assertIsNone(res)

    @patch('dashboard.app.os.path.exists')
    def test_load_csv_artifact_missing(self, mock_exists):
        mock_exists.return_value = False
        res = dashboard_app.load_csv_artifact("missing.csv")
        self.assertIsNone(res)

    # Architectural checks
    def test_dashboard_architecture_is_pure_client(self):
        """
        Scan the dashboard source code to ensure it strictly acts as a client
        and does not import or implement heavy backend logic.
        """
        app_path = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'app.py')
        with open(app_path, 'r') as f:
            content = f.read()
            
        forbidden_terms = [
            "joblib.load",
            "LogisticRegression",
            "predict_proba",
            "sqlite3.connect",
            "from sklearn"
        ]
        
        for term in forbidden_terms:
            self.assertNotIn(term, content, f"Dashboard MUST NOT contain '{term}'. It is a presentation client only.")

if __name__ == '__main__':
    unittest.main()
