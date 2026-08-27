import unittest
import sys
import os
import math
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from decision_policy import decide, process_decision, validate_score, validate_policy
from reason_codes import generate_reasons

class TestDecisionPolicy(unittest.TestCase):
    
    def setUp(self):
        self.valid_policy = {
            "review_threshold": 0.20,
            "hold_threshold": 0.69,
            "policy_version": "1.1"
        }
        
    def test_decision_boundaries(self):
        self.assertEqual(decide(0.19, self.valid_policy), "APPROVE")
        self.assertEqual(decide(0.0, self.valid_policy), "APPROVE")
        self.assertEqual(decide(0.20, self.valid_policy), "REVIEW")
        self.assertEqual(decide(0.50, self.valid_policy), "REVIEW")
        self.assertEqual(decide(0.69, self.valid_policy), "HOLD")
        self.assertEqual(decide(0.99, self.valid_policy), "HOLD")
        self.assertEqual(decide(1.0, self.valid_policy), "HOLD")

    def test_invalid_scores(self):
        invalid_scores = [
            -0.01, 1.01, float('nan'), float('inf'), float('-inf'), "not_a_number", None
        ]
        for score in invalid_scores:
            with self.assertRaises(ValueError):
                validate_score(score)

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            validate_policy({"review_threshold": 0.2})
        with self.assertRaises(ValueError):
            validate_policy({"review_threshold": -0.1, "hold_threshold": 0.69})
        with self.assertRaises(ValueError):
            validate_policy({"review_threshold": 0.2, "hold_threshold": 1.1})
        with self.assertRaises(ValueError):
            validate_policy({"review_threshold": 0.70, "hold_threshold": 0.69})
        with self.assertRaises(ValueError):
            validate_policy({"review_threshold": 0.50, "hold_threshold": 0.50})

    def test_reason_codes(self):
        # We need the model for the new explanation functionality
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'return_risk_model.joblib'))
        if not os.path.exists(model_path):
            self.skipTest("Model file not found for testing reason codes.")
            
        model = joblib.load(model_path)
        
        features = {
            "amount_inr": 15000.0,
            "method": "cod",
            "category": "electronics",
            "is_new_customer": 1,
            "past_orders": 2,
            "past_return_rate": 0.25,
            "order_hour": 2,
            "is_weekend": 0,
            "is_late_night": 1,
            "delivery_distance_km": 60,
            "checkout_time_sec": 5
        }
        
        result = process_decision(0.75, features, self.valid_policy, model)
        
        # 1. Deterministic and Uses Fitted Model
        self.assertGreater(len(result["top_positive_model_contributions"]), 0)
        
        # 2. Positive vs Negative and Top-N / Materiality filtering
        from reason_codes import EXPLANATION_MIN_ABS_CONTRIBUTION
        self.assertLessEqual(len(result["top_positive_model_contributions"]), 3)
        self.assertLessEqual(len(result["top_negative_model_contributions"]), 3)
        
        for c in result["top_positive_model_contributions"]:
            self.assertEqual(c["direction"], "increases_risk")
            self.assertGreaterEqual(c["contribution"], EXPLANATION_MIN_ABS_CONTRIBUTION)
            
        for c in result["top_negative_model_contributions"]:
            self.assertEqual(c["direction"], "reduces_risk")
            self.assertLessEqual(c["contribution"], -EXPLANATION_MIN_ABS_CONTRIBUTION)
            
        # Ensure flat reason codes only surface the top N contributors + domain + base score
        expected_reason_code_count = 1 + len(result["top_positive_model_contributions"]) + len(result["top_negative_model_contributions"]) + len(result["domain_signals"])
        self.assertEqual(len(result["reason_codes"]), expected_reason_code_count)
            
        # 3. Categorical Aggregated (method should be aggregated instead of method_cod)
        features_explained = [c["feature"] for c in result["top_positive_model_contributions"] + result["top_negative_model_contributions"]]
        self.assertNotIn("method_cod", features_explained)
        
        # 4. Reason codes do not claim causality
        texts = [c["reason_text"].lower() for c in result["top_positive_model_contributions"] + result["top_negative_model_contributions"]]
        for t in texts:
            self.assertNotIn("caused", t)
            self.assertNotIn("because", t)
            
        # 5. Domain context
        self.assertIn("DOMAIN_SIGNAL_HIGH_ORDER_VALUE", result["domain_signals"])
        self.assertIn("DOMAIN_SIGNAL_LONG_DELIVERY_DISTANCE", result["domain_signals"])
        
        # 6. Does not override decision
        self.assertEqual(result["decision"], "HOLD")

    def test_decision_context_no_model(self):
        features = {"amount_inr": 500}
        result = process_decision(0.15, features, self.valid_policy, None)
        
        self.assertEqual(result['decision'], 'APPROVE')
        self.assertEqual(result['risk_score'], 0.15)
        self.assertIn("LOW_MODEL_RISK", result['risk_score_reason'])
        self.assertEqual(result['review_threshold'], 0.20)

    def test_model_reconstruction_and_aggregation(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'return_risk_model.joblib'))
        if not os.path.exists(model_path):
            self.skipTest("Model file not found")
        model = joblib.load(model_path)
        
        features = {
            "amount_inr": 25000.0,
            "method": "netbanking",
            "category": "apparel",
            "is_new_customer": 1,
            "past_orders": 0,
            "past_return_rate": 0.35,
            "order_hour": 2,
            "is_weekend": 1,
            "is_late_night": 1,
            "delivery_distance_km": 80,
            "checkout_time_sec": 5
        }
        
        import pandas as pd
        from scipy.special import expit
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        X_transformed = preprocessor.transform(pd.DataFrame([features]))
        if hasattr(X_transformed, 'toarray'):
            X_arr = X_transformed.toarray()[0]
        else:
            X_arr = X_transformed[0]
            
        sum_wx = sum(X_arr * classifier.coef_[0])
        logit = sum_wx + classifier.intercept_[0]
        prob = expit(logit)
        actual_prob = classifier.predict_proba(X_transformed)[0, 1]
        
        self.assertAlmostEqual(prob, actual_prob, places=6)
        
        from reason_codes import extract_model_contributions
        contributions = extract_model_contributions(model, features)
        method_contrib = next(c for c in contributions if c['feature'] == 'method')
        
        from pipeline_utils import CATEGORICAL_FEATURES
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
        idx = list(cat_names).index('method_netbanking')
        numeric_len = len(preprocessor.named_transformers_['num'].get_feature_names_out())
        netbanking_coef = classifier.coef_[0][numeric_len + idx]
        
        self.assertAlmostEqual(method_contrib['contribution'], netbanking_coef * 1.0, places=6)

if __name__ == '__main__':
    unittest.main()
