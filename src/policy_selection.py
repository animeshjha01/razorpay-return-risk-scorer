import pandas as pd
import numpy as np
import json
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, precision_score, recall_score
from pipeline_utils import get_model, FEATURES

def evaluate_policy(y_true, y_prob, amount, t_rev, t_hold, review_cost, hold_cost, review_residual_risk_rate=0.0):
    """
    Evaluates a single threshold pair and calculates all relevant rates and costs.
    """
    preds = np.where(y_prob >= t_hold, 'HOLD', 
                     np.where(y_prob >= t_rev, 'REVIEW', 'APPROVE'))
    
    n_total = len(y_true)
    n_approve = np.sum(preds == 'APPROVE')
    n_review = np.sum(preds == 'REVIEW')
    n_hold = np.sum(preds == 'HOLD')
    
    legit_mask = (y_true == 0)
    risky_mask = (y_true == 1)
    
    tp = np.sum(risky_mask & (preds != 'APPROVE'))
    total_risky = np.sum(risky_mask)
    recall = tp / total_risky if total_risky > 0 else 0.0
    
    total_interventions = n_review + n_hold
    precision = tp / total_interventions if total_interventions > 0 else 0.0
    
    # Costs
    missed_risk_cost = np.sum(amount[risky_mask & (preds == 'APPROVE')])
    review_residual_cost = np.sum(amount[risky_mask & (preds == 'REVIEW')]) * review_residual_risk_rate
    total_review_cost = (n_review * review_cost) + review_residual_cost
    total_hold_cost = n_hold * hold_cost
    total_est_cost = missed_risk_cost + total_review_cost + total_hold_cost
    
    return {
        'review_threshold': float(round(t_rev, 2)),
        'hold_threshold': float(round(t_hold, 2)),
        'approval_rate': n_approve / n_total,
        'review_rate': n_review / n_total,
        'hold_rate': n_hold / n_total,
        'total_intervention_rate': total_interventions / n_total,
        'risky_order_recall': recall,
        'intervention_precision': precision,
        'false_positive_count': int(np.sum(legit_mask & (preds != 'APPROVE'))),
        'false_negative_count': int(np.sum(risky_mask & (preds == 'APPROVE'))),
        'review_cost': float(total_review_cost),
        'hold_cost': float(total_hold_cost),
        'missed_risk_cost': float(missed_risk_cost),
        'total_estimated_cost': float(total_est_cost)
    }

def grid_search_policy(y_true, y_prob, amount, review_cost, hold_cost, max_intervention, max_hold, review_residual_risk_rate=0.0):
    """
    Searches the T_review < T_hold grid, filtering by constraints, minimizing cost, applying tie-breakers.
    """
    candidates = []
    
    # 0.01 resolution grid
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    for t_rev in thresholds:
        for t_hold in thresholds:
            if t_rev >= t_hold:
                continue
            
            metrics = evaluate_policy(y_true, y_prob, amount, t_rev, t_hold, review_cost, hold_cost, review_residual_risk_rate)
            
            feasible = True
            rejection_reason = "N/A"
            if metrics['total_intervention_rate'] > max_intervention:
                feasible = False
                rejection_reason = "Violates MAX_INTERVENTION_RATE"
            elif metrics['hold_rate'] > max_hold:
                feasible = False
                rejection_reason = "Violates MAX_HOLD_RATE"
                
            metrics['feasibility_status'] = "Feasible" if feasible else "Infeasible"
            metrics['rejection_reason'] = rejection_reason
            candidates.append(metrics)
            
    feasible_cands = [c for c in candidates if c['feasibility_status'] == 'Feasible']
    
    if not feasible_cands:
        return None, candidates
        
    # Find minimum cost
    min_cost = min(c['total_estimated_cost'] for c in feasible_cands)
    
    # Tie-breaker logic (within a margin of 10 INR)
    tied = [c for c in feasible_cands if c['total_estimated_cost'] <= min_cost + 10.0]
    
    # Sort by: 1) Hold rate (asc), 2) Recall (desc -> negative asc), 3) Intervention rate (asc)
    tied.sort(key=lambda x: (x['hold_rate'], -x['risky_order_recall'], x['total_intervention_rate']))
    
    return tied[0], candidates

def run_sensitivity_analysis(y_true, y_prob, amount):
    """
    Runs scenarios with different economic costs, capacities, and review leakages.
    """
    scenarios = [
        {"name": "A (Low Friction)", "rev": 10, "hold": 50, "max_int": 0.25, "max_hld": 0.05, "leak": 0.0},
        {"name": "B (Primary Baseline)", "rev": 50, "hold": 150, "max_int": 0.25, "max_hld": 0.05, "leak": 0.10},
        {"name": "C (High Friction)", "rev": 150, "hold": 500, "max_int": 0.25, "max_hld": 0.05, "leak": 0.0},
        {"name": "Cap 5% Intervention", "rev": 50, "hold": 150, "max_int": 0.05, "max_hld": 0.05, "leak": 0.0},
        {"name": "Cap 10% Intervention", "rev": 50, "hold": 150, "max_int": 0.10, "max_hld": 0.05, "leak": 0.0},
        {"name": "Cap 15% Intervention", "rev": 50, "hold": 150, "max_int": 0.15, "max_hld": 0.05, "leak": 0.0},
        {"name": "Leakage 5%", "rev": 50, "hold": 150, "max_int": 0.25, "max_hld": 0.05, "leak": 0.05},
        {"name": "Leakage 10%", "rev": 50, "hold": 150, "max_int": 0.25, "max_hld": 0.05, "leak": 0.10},
        {"name": "Leakage 20%", "rev": 50, "hold": 150, "max_int": 0.25, "max_hld": 0.05, "leak": 0.20},
    ]
    
    results = []
    print("\n--- SENSITIVITY ANALYSIS ---")
    for s in scenarios:
        best, _ = grid_search_policy(y_true, y_prob, amount, s['rev'], s['hold'], s['max_int'], s['max_hld'], s['leak'])
        if best is None:
            print(f"Scenario: {s['name']:25} -> NO FEASIBLE POLICY (Check operational constraints)")
            results.append({"scenario": s['name'], "status": "INFEASIBLE"})
        else:
            print(f"Scenario: {s['name']:25} | T_rev: {best['review_threshold']:.2f} | T_hold: {best['hold_threshold']:.2f} | "
                  f"RevRate: {best['review_rate']:.1%} | HldRate: {best['hold_rate']:.1%} | "
                  f"Cost: INR {best['total_estimated_cost']:,.0f} | Recall: {best['risky_order_recall']:.1%}")
            results.append({
                "scenario": s['name'], "status": "FEASIBLE",
                "T_rev": best['review_threshold'], "T_hold": best['hold_threshold'],
                "Cost": best['total_estimated_cost']
            })
    return results

def main(train_csv_path='data/train.csv'):
    print(f"Loading ONLY training data from: {train_csv_path}")
    # Strictly isolated from test data. 
    # Purpose: Select policy using internal validation split.
    
    df = pd.read_csv(train_csv_path)
    
    # Internal Split
    train_internal, val_internal = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['returned_or_chargeback']
    )
    
    X_train = train_internal[FEATURES]
    y_train = train_internal['returned_or_chargeback']
    
    X_val = val_internal[FEATURES]
    y_val = val_internal['returned_or_chargeback']
    amount_val = val_internal['amount_inr'].values
    
    print("Fitting model on internal training split...")
    model = get_model('lr')
    model.fit(X_train, y_train)
    
    print("Generating validation probabilities...")
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Calibration Check
    brier = brier_score_loss(y_val, y_prob)
    print(f"\nCalibration Diagnostic (Brier Score): {brier:.4f}")
    if brier < 0.15:
        print("-> Calibration is acceptable (well-calibrated). Using original probabilities.")
    else:
        print("-> WARNING: Calibration is poor. However, no auto-calibration will be applied to prevent complexity.")
        
    # Primary Assumptions
    REV_COST = 50
    HOLD_COST = 150
    MAX_INT_RATE = 0.25
    MAX_HLD_RATE = 0.05
    REV_RESIDUAL_RISK = 0.10 # 10% leakage assumption by default
    
    print("\nEvaluating Candidate Policy Grid...")
    best_policy, all_candidates = grid_search_policy(
        y_val, y_prob, amount_val, REV_COST, HOLD_COST, MAX_INT_RATE, MAX_HLD_RATE, REV_RESIDUAL_RISK
    )
    
    if best_policy is None:
        raise ValueError("Primary baseline assumptions yielded NO FEASIBLE POLICY. Please review constraints.")
        
    print("\n--- PRIMARY OPTIMAL POLICY ---")
    print(f"Review Threshold: {best_policy['review_threshold']}")
    print(f"Hold Threshold: {best_policy['hold_threshold']}")
    print(f"Review Rate: {best_policy['review_rate']:.1%}")
    print(f"Hold Rate: {best_policy['hold_rate']:.1%}")
    print(f"Total Intervention Rate: {best_policy['total_intervention_rate']:.1%}")
    print(f"Risk Recall: {best_policy['risky_order_recall']:.1%}")
    print(f"Total Estimated Cost: INR {best_policy['total_estimated_cost']:,.0f}")
    
    # Run Sensitivity Analysis
    sensitivity_results = run_sensitivity_analysis(y_val, y_prob, amount_val)
    
    # Save Artifacts
    config = {
        "model_type": "LogisticRegression (Unweighted)",
        "model_version": "LR_UNWEIGHTED_V1",
        "policy_version": "1.1",
        "validation_seed": 42,
        "review_threshold": best_policy['review_threshold'],
        "hold_threshold": best_policy['hold_threshold'],
        "review_cost_inr": REV_COST,
        "hold_cost_inr": HOLD_COST,
        "review_residual_risk_rate": REV_RESIDUAL_RISK,
        "false_negative_cost_definition": "amount_inr",
        "max_intervention_rate": MAX_INT_RATE,
        "max_hold_rate": MAX_HLD_RATE,
        "selection_methodology": "Minimize total unconditionally-incurred friction cost + FN risk loss, subject to capacity constraints. Tie-breakers: min hold rate, max recall, min intervention.",
        "calibration_metric": f"Brier Score = {brier:.4f}",
        "validation_statistics": best_policy,
        "selection_scenario": "Primary Baseline Simulation",
        "selection_timestamp": "2026-01-01T12:00:00+00:00"
    }
    
    # Note: These optimal thresholds are highly sensitive to the synthetic business assumptions defined above,
    # and are not universal mathematical facts. They represent a business decision boundary.
    
    with open('models/policy_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    pd.DataFrame(all_candidates).to_csv('models/policy_candidates.csv', index=False)
    print("\nSaved models/policy_config.json and models/policy_candidates.csv")

if __name__ == "__main__":
    main()
