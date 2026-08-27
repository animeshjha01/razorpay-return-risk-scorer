import pandas as pd
import numpy as np
import json
import joblib
from pipeline_utils import get_model, FEATURES
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def train():
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    X_train = train_df[FEATURES]
    y_train = train_df['returned_or_chargeback']
    
    X_test = test_df[FEATURES]
    y_test = test_df['returned_or_chargeback']
    
    # We use Logistic Regression as it outperformed HGB in Phase 1
    print("Training final Logistic Regression model on FULL training set...")
    model = get_model('lr')
    model.fit(X_train, y_train)
    
    # Save the final model
    joblib.dump(model, 'models/return_risk_model.joblib')
    
    # Load Frozen Policy
    with open('models/policy_config.json', 'r') as f:
        policy = json.load(f)
        
    t_rev = policy['review_threshold']
    t_hold = policy['hold_threshold']
    review_cost_rate = policy['review_cost_inr']
    hold_cost_rate = policy['hold_cost_inr']
    
    print("\n--- FINAL HELD-OUT TEST EVALUATION ---")
    print(f"Applying frozen policy (T_rev={t_rev}, T_hold={t_hold})...")
    
    y_prob = model.predict_proba(X_test)[:, 1]
    amount = test_df['amount_inr'].values
    # Apply authoritative policy cost function
    from policy_selection import evaluate_policy
    
    metrics = evaluate_policy(
        y_test.values, 
        y_prob, 
        amount, 
        t_rev, 
        t_hold, 
        review_cost_rate, 
        hold_cost_rate, 
        policy.get('review_residual_risk_rate', 0.0)
    )
    
    print(f"Approval Rate: {metrics['approval_rate']:.1%}")
    print(f"Review Rate: {metrics['review_rate']:.1%}")
    print(f"Hold Rate: {metrics['hold_rate']:.1%}")
    print(f"Risk Recall: {metrics['risky_order_recall']:.1%}")
    print(f"Intervention Precision: {metrics['intervention_precision']:.1%}")
    print(f"Total Test Cost: INR {metrics['total_estimated_cost']:,.0f}")
    
    # Save the final results to json
    final_results = {
        "approval_rate": metrics['approval_rate'],
        "review_rate": metrics['review_rate'],
        "hold_rate": metrics['hold_rate'],
        "risk_recall": metrics['risky_order_recall'],
        "intervention_precision": metrics['intervention_precision'],
        "missed_risk_cost": metrics['missed_risk_cost'],
        "total_review_cost": metrics['review_cost'],
        "total_hold_cost": metrics['hold_cost'],
        "total_estimated_cost": metrics['total_estimated_cost']
    }
    with open('models/test_evaluation_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print("\n--- FINAL HELD-OUT TEST DIAGNOSTICS ---")
    print("(REPORTING ONLY - THESE DO NOT INFLUENCE POLICY SELECTION)")
    
    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"Test ROC-AUC: {roc:.4f}")
    print(f"Test Average Precision: {ap:.4f}")
    
    thresholds = [0.2, 0.4, 0.6, 0.8]
    for t in thresholds:
        binary_preds = (y_prob >= t).astype(int)
        prec = precision_score(y_test, binary_preds, zero_division=0)
        rec = recall_score(y_test, binary_preds, zero_division=0)
        f1 = f1_score(y_test, binary_preds, zero_division=0)
        print(f"Thr {t}: Prec: {prec:.1%}, Rec: {rec:.1%}, F1: {f1:.3f}")
        
    print("Training complete. Artifacts saved in models/")

if __name__ == "__main__":
    train()
