import pandas as pd
import numpy as np
import json
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

def train():
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    features = [
        'amount_inr', 'method', 'category', 'is_new_customer', 
        'past_orders', 'past_return_rate', 'order_hour', 
        'is_weekend', 'is_late_night', 'delivery_distance_km', 
        'checkout_time_sec'
    ]
    
    X_train = train_df[features]
    y_train = train_df['returned_or_chargeback']
    
    X_test = test_df[features]
    y_test = test_df['returned_or_chargeback']
    
    numeric_features = [
        'amount_inr', 'past_orders', 'past_return_rate', 
        'order_hour', 'delivery_distance_km', 'checkout_time_sec'
    ]
    categorical_features = ['method', 'category']
    passthrough_features = ['is_new_customer', 'is_weekend', 'is_late_night']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ]
    )
    
    print("Training Logistic Regression baseline...")
    lr_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])
    lr_model.fit(X_train, y_train)
    
    print("Training regularized HistGradientBoostingClassifier...")
    # Using L2 regularization, shallow depth, min_samples_leaf to prevent overfitting
    gb_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            max_depth=5,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            class_weight='balanced',
            random_state=42
        ))
    ])
    gb_model.fit(X_train, y_train)
    
    print("Evaluating models...")
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    gb_probs = gb_model.predict_proba(X_test)[:, 1]
    
    lr_roc = roc_auc_score(y_test, lr_probs)
    lr_ap = average_precision_score(y_test, lr_probs)
    
    gb_roc = roc_auc_score(y_test, gb_probs)
    gb_ap = average_precision_score(y_test, gb_probs)
    
    comparison = {
        'LogisticRegression': {
            'ROC_AUC': lr_roc,
            'AveragePrecision': lr_ap
        },
        'HistGradientBoosting': {
            'ROC_AUC': gb_roc,
            'AveragePrecision': gb_ap
        }
    }
    
    with open('models/model_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=4)
        
    print(f"LR  - ROC-AUC: {lr_roc:.4f}, AP: {lr_ap:.4f}")
    print(f"HGB - ROC-AUC: {gb_roc:.4f}, AP: {gb_ap:.4f}")
    
    primary_model = gb_model if gb_roc >= lr_roc else lr_model
    primary_probs = gb_probs if gb_roc >= lr_roc else lr_probs
    print(f"Primary model chosen based on ROC-AUC: {'HistGradientBoosting' if gb_roc >= lr_roc else 'LogisticRegression'}")
    
    joblib.dump(primary_model, 'models/return_risk_model.joblib')
    
    # Threshold analysis
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    metrics = []
    costs = []
    
    # FP cost: 50 INR, FN cost: actual order amount
    fp_cost_rate = 50.0
    
    for t in thresholds:
        preds = (primary_probs >= t).astype(int)
        
        metrics.append({
            'threshold': t,
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0),
            'tp': int(np.sum((y_test == 1) & (preds == 1))),
            'fp': int(np.sum((y_test == 0) & (preds == 1))),
            'tn': int(np.sum((y_test == 0) & (preds == 0))),
            'fn': int(np.sum((y_test == 1) & (preds == 0)))
        })
        
        # Calculate cost
        false_positives_mask = (y_test == 0) & (preds == 1)
        false_negatives_mask = (y_test == 1) & (preds == 0)
        
        cost_fp = np.sum(false_positives_mask) * fp_cost_rate
        cost_fn = test_df.loc[false_negatives_mask, 'amount_inr'].sum()
        
        costs.append({
            'threshold': t,
            'fp_cost': float(cost_fp),
            'fn_cost': float(cost_fn),
            'total_cost': float(cost_fp + cost_fn)
        })
        
    pd.DataFrame(metrics).to_csv('models/metrics_by_threshold.csv', index=False)
    pd.DataFrame(costs).to_csv('models/cost_by_threshold.csv', index=False)
    
    # Feature importance
    print("Calculating permutation importance...")
    result = permutation_importance(primary_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1, scoring='roc_auc')
    importance_df = pd.DataFrame({
        'feature': features,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    importance_df.to_csv('models/feature_importance.csv', index=False)
    print("Training complete. Artifacts saved in models/")

if __name__ == "__main__":
    train()
