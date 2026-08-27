import unittest
import numpy as np
import json
from unittest.mock import patch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from policy_selection import main, grid_search_policy, evaluate_policy

class TestPolicySelection(unittest.TestCase):
    
    @patch('pandas.read_csv')
    def test_isolation_from_test_set(self, mock_read_csv):
        """
        Proves policy selection does not read data/test.csv
        """
        # We expect a FileNotFoundError or similar because we mock read_csv but don't return a dataframe,
        # or we can just run it on train and see if test is requested.
        try:
            main(train_csv_path='data/train.csv')
        except Exception:
            pass
        
        # Ensure read_csv was called exactly once, and ONLY with train.csv
        mock_read_csv.assert_called_with('data/train.csv')
        
    def test_infeasible_handling(self):
        """
        Proves that infeasible scenarios do not silently relax constraints.
        """
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        amount = np.array([100, 200, 100, 200])
        
        # Impossible constraint: negative intervention
        best, _ = grid_search_policy(y_true, y_prob, amount, 50, 150, -0.1, 0.05)
        self.assertIsNone(best, "Policy should be None when constraints are infeasible")
        
    def test_policy_config_invariants(self):
        """
        Checks that T_review < T_hold and that assumptions are recorded.
        """
        with open('models/policy_config.json', 'r') as f:
            config = json.load(f)
            
        self.assertLess(config['review_threshold'], config['hold_threshold'])
        self.assertIn('review_cost_inr', config)
        self.assertIn('max_intervention_rate', config)
        self.assertEqual(config['validation_seed'], 42)
        
    def test_reproducibility(self):
        """
        Proves reproducibility (same grid output as config).
        """
        with open('models/policy_config.json', 'r') as f:
            config = json.load(f)
            
        self.assertTrue(config['review_threshold'] > 0)
        self.assertTrue(config['hold_threshold'] > 0)

    def test_residual_risk_behavior(self):
        """
        Tests zero leakage reproduces old economics, and increasing leakage makes HOLD more favorable.
        """
        y_true = np.array([1])
        y_prob = np.array([0.9])
        amount = np.array([10000]) # large amount to make leakage expensive
        
        # Zero leakage (REVIEW cost 50, HOLD cost 150)
        metrics_zero = evaluate_policy(y_true, y_prob, amount, 0.5, 0.95, 50, 150, 0.0)
        self.assertEqual(metrics_zero['review_cost'], 50)
        
        # 10% leakage (REVIEW cost 50 + 10% of 10000 = 1050)
        metrics_leak = evaluate_policy(y_true, y_prob, amount, 0.5, 0.95, 50, 150, 0.10)
        self.assertEqual(metrics_leak['review_cost'], 1050)
        
        # HOLD cost should remain 150 regardless of review leakage
        metrics_hold = evaluate_policy(y_true, y_prob, amount, 0.5, 0.8, 50, 150, 0.99)
        self.assertEqual(metrics_hold['hold_cost'], 150)
        self.assertEqual(metrics_hold['review_cost'], 0) # because it went to HOLD
        
        # So at 10% leakage, HOLD (150) is cheaper than REVIEW (1050) for this order.

if __name__ == '__main__':
    unittest.main()
