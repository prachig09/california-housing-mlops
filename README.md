# Automated MLOps Continuous Training & Delivery Pipeline

An end-to-end MLOps CI/CD pipeline built with **Python**, **scikit-learn**, **pytest**, and **GitHub Actions**. Automatically validates input schema, fits preprocessing, trains candidate ML models against a dummy baseline, executes unit/schema tests, and publishes versioned model packages upon passing quality gates.

## Dataset & Task Overview
* **Dataset:** California Housing Dataset (Scikit-Learn / US Census Data)
* **Target Variable:** `MedHouseVal` (Median house value for California districts, expressed in $100,000s)
* **Input Features:** `MedInc` (Median income), `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.
* **Prediction Task:** Supervised Regression predicting block-level real estate valuations.

---

## Technical Specifications

### Evaluation Metric Selection
**Mean Squared Error (MSE)** was chosen as the primary evaluation metric. Real estate valuation demands strict penalties on high-variance outlier errors. MSE penalizes larger deviations quadratically, discouraging model drift on high-value properties.

### Improvement Margin Justification
* **Baseline Score (`DummyRegressor` predicting median):** ~1.32 MSE.
* **Margin Defined:** **0.50 MSE** ($50,000² scale reduction).
* **Quality Rule:** Candidate MSE must satisfy `candidate_mse <= baseline_mse - 0.50`.
* **Trade-off Analysis:** Setting the margin too low (< 0.05) risks releasing underfitted models affected by sampling noise. Setting the margin excessively high (> 0.90) rejects robust models that offer practical operational value.

---

## CI/CD Pipeline Failure & Recovery Demonstration

1. **Failure A - Model Quality Gate Breach:** 
   * **Run Link:** https://github.com/prachig09/california-housing-mlops/actions/runs/35986229225/job/107589421589
   * **Cause:** Trained an underfit 1-tree `RandomForestRegressor` (`max_depth=1`). 
   * **Outcome:** Quality gate check intercepted performance regression (`candidate_mse` exceeded threshold). The step exited via `sys.exit(1)` and blocked package publication.
2. **Failure B - Application Test Breach:** 
   * **Run Link:** `[PASTE_LINK_TO_FAILED_RUN_2]`
   * **Cause:** Injected schema test assertion error in `tests/test_app.py`.
   * **Outcome:** `pytest` caught the broken contract test and terminated execution, preventing deployment of a broken prediction interface.
3. **Final Successful Pipeline:** 
   * **Run Link:** `[PASTE_LINK_TO_SUCCESSFUL_RUN]`
   * **Artifact Package:** `california-housing-model-run-3-commit-[COMMIT_HASH]`

---

## MLOps Maturity Assessment
* **Current Level:** **MLOps Level 1 (Automated Pipeline Execution)**
  * *Capabilities:* Automated schema validation, training automation, automated contract/unit testing, metric-gated artifact publishing, and versioned artifact generation on code push.
* **Path to MLOps Level 2 (Continuous Training & Deployment):**
  * Implement automated data drift detection (e.g., Evidently AI / Great Expectations).
  * Integrate continuous feature store (e.g., Feast) and live target tracking.
  * Automate containerized microservice deployments (Docker/Kubernetes) upon artifact upload.