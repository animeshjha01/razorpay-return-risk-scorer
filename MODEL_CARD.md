# Model Card

## Model Identity
* **Model Type:** LogisticRegression
* **Model Version:** `LR_UNWEIGHTED_V1`

## Intended Use
* **Primary Use:** Return / chargeback risk ranking and decision support for e-commerce orders.
* **Paradigm:** Defense-only capability designed to intercept high-risk behavior.
* **Scope:** Synthetic demonstration for hackathon evaluation purposes.

## Out-of-Scope Use
* Autonomous payment blocking without human oversight.
* Real production risk decisions on live payment traffic.
* Creditworthiness or lending decisions.
* Customer profiling or behavioral scoring beyond this explicit transaction-risk use case.

## Dataset
The dataset is entirely synthetic, created to demonstrate the decisioning architecture.
* **Total Size:** 5,000 records
* **Splits:** Train = 4,000 records, Final Held-Out Test = 1,000 records
* **Positive Rate:** Approximately 16.4% target prevalence

**Synthetic Domain Constraints:**
The data generator enforces strict domain boundaries to mimic an e-commerce environment (e.g., Cash on Delivery is capped at ₹30,000; digital products mask delivery distances).

## Features
The model consumes the following explicit features:
* `amount_inr` (Numeric)
* `method` (Categorical)
* `category` (Categorical)
* `is_new_customer` (Binary flag)
* `past_orders` (Numeric)
* `past_return_rate` (Numeric ratio 0.0-1.0)
* `order_hour` (Numeric 0-23)
* `is_weekend` (Binary flag)
* `is_late_night` (Binary flag)
* `delivery_distance_km` (Numeric)
* `checkout_time_sec` (Numeric)

## Preprocessing
* **Numeric Features:** Transformed via `StandardScaler`.
* **Categorical Features:** Transformed via `OneHotEncoder(handle_unknown="ignore")`.
* **Binary Features:** Passed through without modification.
These transformations are bundled into a unified scikit-learn `Pipeline`.

## Training Procedure
1. The pipeline is initially trained on an internal 80% split of the training data.
2. The remaining 20% validation split is used exclusively for evaluating thresholds and selecting the cost-aware policy.
3. After the policy is frozen, the model undergoes a final retraining pass over the complete 100% training dataset to maximize data efficiency.

## Calibration
The model produces reasonably calibrated probability estimates on the internal validation split, with a Brier Score of 0.1218 versus 0.1369 for the prevalence baseline. No secondary calibration layer is applied.

*Note: The Brier score is a validation-only diagnostic. It was not optimized directly during training, and the model output remains subject to the statistical uncertainty and limitations of the synthetic data distribution.*

## Performance

The evaluation metrics are explicitly separated to prevent conflating internal diagnostics with final held-out test performance.

### Final Held-Out Test
* **ROC-AUC:** 0.759
* **Average Precision (PR-AUC):** 0.404

### Validation Diagnostics
* **Validation Brier Score:** 0.1218
* **Validation Prevalence:** 16.38%
* **Naive Baseline Brier Score:** 0.1369

## Decision Policy
The model is paired with a deterministic operational policy based on simulated economics:
* **Thresholds:** `0.20` (Review), `0.69` (Hold).
* **Costs:** ₹50 (Review intervention), ₹150 (Hold intervention).
* **Constraints:** Maximum 25% intervention capacity, maximum 5% hold capacity.
* **Residual Risk:** Assumes manual review fails to catch 10% of true risk.
* **Sensitivity:** Selected by simulating a 100x100 resolution grid of thresholds over the validation dataset to minimize total incurred cost.

## Explainability
Model decisions are explained using direct coefficient math:
`contribution_j = coefficient_j × transformed_feature_j`

These contributions are interpreted in log-odds space. Because this represents the internal mathematical state of the Logistic Regression model, the explanations are strictly model-consistent. However, they are **not causal** inferences about the physical world.

## Fairness / Limitations
The dataset is synthetic. Therefore, it does not support any real-world demographic fairness conclusions. A true production fairness evaluation would require appropriate real-world transaction data, sensitive attribute masking, and formal bias governance. 

## Monitoring / Drift
* No live data drift monitoring is implemented in this MVP.
* The synthetic environment is mathematically stationary.
* Production deployments would strictly require continuous distribution drift detection and calibration monitoring.

## Security / Privacy
* There are no secrets, API keys, or actual user credentials in this repository.
* The synthetic data contains no real Personally Identifiable Information (PII) or payment credentials.
* The audit database is local-only and is not cryptographically immutable.

## Known Modeling Limitation
Due to the artificial constraints applied during synthetic data generation (e.g., hard-capping Cash on Delivery at ₹30,000), `amount_inr` and `method` exhibit non-intuitive covariance. This results in the Logistic Regression model occasionally assigning counter-intuitive coefficient weights to the `amount_inr` feature. This is a recognized limitation of the synthetic generation logic rather than an algorithmic bug.
