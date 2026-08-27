# Architecture

This document describes the end-to-end technical architecture of the Razorpay Return Risk Scorer project.

## System Overview

The system is decoupled into distinct architectural layers to separate concerns, enforce immutability of audited records, and provide clear operational boundaries.

### 1. Presentation
* **Component:** Streamlit Dashboard
* **Role:** A judge-facing client for testing order combinations, viewing real-time scoring insights, browsing historical audit logs, and reviewing policy analytics.
* **Boundary:** Does not calculate risk or implement model logic. Communicates exclusively via HTTP.

### 2. Transport
* **Component:** FastAPI
* **Role:** Exposes synchronous HTTP endpoints.
* **Boundary:** Handles Pydantic input validation, HTTP exception mapping, and response serialization. Does not implement ML math.

### 3. Orchestration
* **Component:** ScoringService
* **Role:** The core domain logic coordinator.
* **Boundary:** Wires together the pre-loaded ML model, frozen policy, explainability engine, and audit database.

### 4. ML
* **Component:** Logistic Regression + Shared Preprocessing
* **Role:** Converts raw order features into a scaled risk probability.
* **Boundary:** Contains `StandardScaler` and `OneHotEncoder`. Outputs probabilities. Does not make business decisions.

### 5. Policy
* **Component:** Frozen cost-aware thresholds
* **Role:** Evaluates the risk probability against pre-calculated boundary thresholds.
* **Boundary:** Purely deterministic. Maps probabilities to `APPROVE`, `REVIEW`, or `HOLD`.

### 6. Explanation
* **Component:** Log-odds coefficient contributions
* **Role:** Calculates the exact mathematical contribution of each feature to the final risk score.
* **Boundary:** Read-only analysis of model coefficients.

### 7. Audit
* **Component:** SQLite Append-only Application Interface
* **Role:** Durably stores every decision processed by the scoring service.
* **Boundary:** Application-level append-only. Does not make decisions or alter historical records.

## End-to-End Request Flow

1. **Request:** Client submits a JSON payload to `POST /score-order`.
2. **Pydantic Validation:** The API layer verifies type safety and bounds.
3. **Domain Validation:** The ScoringService verifies strict domain bounds (e.g., `order_hour` in 0-23, `past_return_rate` in 0-1).
4. **Model Prediction:** The model predicts the probability of a return/chargeback.
5. **Policy Decision:** The probability is compared to `T_review` and `T_hold` to determine the intervention.
6. **Explanation:** The explainability engine computes Top 3 Positive and Top 3 Negative feature contributions.
7. **Audit Append:** The complete decision record is written to the SQLite database. If this fails, the transaction aborts (HTTP 503).
8. **Response:** The detailed `ScoringResponse` is returned to the client.

## Data Flow

Data flows strictly one way during the offline model building phase:

1. `train.csv` is loaded and an **internal 80/20 validation split** is created.
2. The initial model is trained on the internal 80% split.
3. The internal 20% validation split is used for **Policy Selection**, calculating the optimal capacity-constrained cost boundaries.
4. `test.csv` is explicitly held out. It is **NOT used during threshold selection**.
5. After the optimal policy is selected and frozen as `policy_config.json`, the model undergoes **final retraining** on the full 100% `train.csv` dataset.
6. **Final evaluation** is performed on the completely unseen `test.csv`, generating `test_evaluation_results.json`.

## Policy-Selection Architecture

The cost-aware policy selector simulates hypothetical business costs across a dense grid of thresholds.
* **Resolution:** Evaluates thresholds from 0.01 to 0.99 in increments of 0.01.
* **Cost Function:** Calculates hypothetical operational costs (Review = ₹50, Hold = ₹150) plus risk-loss costs (incorporating a 10% residual review-risk assumption).
* **Capacity Constraints:** Drops any threshold combination that exceeds the maximum operational limits (25% maximum intervention rate, 5% maximum hold rate).
* **Sensitivity Analysis:** Retains candidate curves for analytics, freezing the mathematically optimal thresholds into the JSON artifact.

## Explainability Architecture

To assist human reviewers without resorting to black-box approximations, the system surfaces exact Logistic Regression mechanics:
* **Transformation:** Raw features are scaled and one-hot encoded.
* **Coefficient Multiplication:** The transformed features are multiplied by the model coefficients.
* **Categorical Aggregation:** Contributions from one-hot encoded categories belonging to the same parent feature are aggregated.
* **Materiality Filter:** Absolute contributions below 0.05 log-odds are suppressed as statistical noise.
* **Top-N Surface Rule:** The system sorts and returns the Top 3 positive (risk-increasing) and Top 3 negative (risk-decreasing) features.

## Audit Architecture

* **Schema:** Flat relational table representing the exact decision state.
* **Append Behavior:** Application code strictly uses `INSERT INTO` operations.
* **Read-Only API:** Endpoints are exclusively `GET` operations.
* **Version Tracking:** Every record stamps the `model_version` and `policy_version`.
* **Timestamp:** Captured dynamically in UTC ISO-8601 format.
* **Failure Semantics:** If the SQLite `INSERT` fails or the database is locked, the API aborts the request and returns a `503 Service Unavailable` error, ensuring no unaudited decisions reach the client.

## Error Handling

* **Malformed request:** 400 Bad Request (handled natively by Pydantic).
* **Domain-invalid request:** 400 Bad Request (handled by ScoringService domain checks).
* **Model failure:** 503 Service Unavailable (e.g., model artifacts missing at startup).
* **Policy failure:** 503 Service Unavailable (e.g., config corrupted).
* **Audit failure:** 503 Service Unavailable (e.g., database lock/IO error).
* **Unexpected server failure:** 500 Internal Server Error (safely masks stack traces).

## Component Boundaries

* The **Dashboard** must not score data directly; it must rely on the API.
* The **API layer** must not implement model math or policy application.
* The **Audit layer** must not evaluate policy or make risk decisions.
* The **Policy Selection** script must not read or process `test.csv`.

## Reproducibility

The system provides a deterministic offline pipeline:
1. `generate_data.py` (CLI arguments for N-size and RNG seed)
2. `policy_selection.py` (Isolates threshold optimization)
3. `train.py` (Evaluates and freezes the final models)

## Scalability Boundary

This architecture employs SQLite as a lightweight MVP storage engine to simplify hackathon deployment. While functional, it relies on file-level locking. Under high-throughput concurrent production loads, write-contention would cause `database is locked` errors. A true production rollout would require migrating the `audit_log.py` implementation to a high-concurrency database (e.g., PostgreSQL) or an event-streaming architecture.
