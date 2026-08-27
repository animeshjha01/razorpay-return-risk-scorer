import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

# Define shared feature lists
FEATURES = [
    'amount_inr', 'method', 'category', 'is_new_customer', 
    'past_orders', 'past_return_rate', 'order_hour', 
    'is_weekend', 'is_late_night', 'delivery_distance_km', 
    'checkout_time_sec'
]

NUMERIC_FEATURES = [
    'amount_inr', 'past_orders', 'past_return_rate', 
    'order_hour', 'delivery_distance_km', 'checkout_time_sec'
]

CATEGORICAL_FEATURES = ['method', 'category']
PASSTHROUGH_FEATURES = ['is_new_customer', 'is_weekend', 'is_late_night']

# Authoritative source of truth for the active model version
MODEL_VERSION = "LR_UNWEIGHTED_V1"

def get_preprocessor():
    """
    Returns the shared ColumnTransformer for preprocessing.
    This guarantees that policy selection and final training use the exact same logic.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('pass', 'passthrough', PASSTHROUGH_FEATURES)
        ]
    )

def get_model(model_type='lr'):
    """
    Returns a complete scikit-learn Pipeline.
    By default uses LogisticRegression, which was chosen during Phase 1 evaluation.
    """
    preprocessor = get_preprocessor()
    if model_type == 'lr':
        classifier = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'hgb':
        classifier = HistGradientBoostingClassifier(
            max_depth=5, min_samples_leaf=20, l2_regularization=1.0, 
            early_stopping=True, class_weight='balanced', random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
