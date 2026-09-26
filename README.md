# Automated MLOps Continuous Training & Delivery Pipeline

An end-to-end MLOps CI/CD pipeline built with **Python**, **scikit-learn**, **pytest**, and **GitHub Actions**. Automatically validates input schema, fits preprocessing, trains candidate ML models against a dummy baseline, executes unit/schema tests, and publishes versioned model packages upon passing quality gates.

---

## Dataset & Task Overview

* **Dataset Source:** California Housing Dataset (Scikit-Learn / 1990 U.S. Census Data, fetched automatically via `sklearn.datasets.fetch_california_housing`).
* **Target Variable:** `MedHouseVal` (Median house value for California block groups, expressed in $100,000s).
* **Input Features (8 tabular numerical features):** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`.
* **Prediction Task:** Supervised Regression predicting block-level real estate valuations.

---

## Repository Structure

```text
california-housing-mlops/
├── .github/
│   └── workflows/
│       └── pipeline.yml          # GitHub Actions CI/CD workflow
├── data/
│   └── .gitkeep                  # Tracks empty data directory in Git
├── src/
│   ├── __init__.py               # Marks src as an importable Python package
│   ├── data.py                   # Data fetching & schema validation logic
│   ├── train.py                  # Model training, baseline evaluation & quality gate
│   └── predict.py                # Inference execution script
├── tests/
│   └── test_app.py               # Pytest suite for model loading & schema validation
├── .gitignore                    # Ignores venv, local cache, and CSV datasets
├── requirements.txt              # Pinned environment dependencies
└── README.md                     # Pipeline documentation and submission report
```
---

## Setup & Local Execution Instructions
1. Environment Setup

    Ensure Python 3.10+ is installed on your local machine.

    ```bash
    # Clone the repository
    git clone [https://github.com/prachig09/california-housing-mlops.git](https://github.com/prachig09/california-housing-mlops.git)
    cd california-housing-mlops
    
    # Create and activate virtual environment
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    
    # Upgrade pip and install dependencies
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

2. Run Pipeline Steps Locally

    Run the following commands in sequence to execute the local pipeline:

    ```bash
    # 1. Fetch dataset and validate schema
    python -m src.data
    
    # 2. Train model and evaluate against baseline quality gate
    python -m src.train
    
    # 3. Test prediction interface
    python -m src.predict
    
    # 4. Run automated test suite
    python -m pytest tests/
    ```
---

## Technical Specifications

### Evaluation Metric Selection

Mean Squared Error (MSE) was chosen as the primary evaluation metric. Real estate valuation demands strict penalties on high-variance outlier errors. Because MSE penalizes larger errors quadratically $(y - \hat{y})^2$, it heavily discourages model drift and large misestimations on high-value properties compared to small, uniform deviations.

Improvement Margin Justification

1. **Baseline Score** (DummyRegressor predicting median): 1.3762 MSE.
2. **Margin Defined:** 0.50 MSE ($50,000² scale error reduction).
3. **Quality Rule:** Candidate MSE must satisfy $\text{Candidate MSE} \le \text{Baseline MSE} - 0.50 \implies \mathbf{\le 0.8762}$.
4. **Model Result:** The candidate Random Forest model achieved an MSE of 0.2732 (reducing average error from ~$117.3k down to ~$52.3k), passing the gate (gate_passed: true).
5. **Trade-off Analysis:** Setting the margin too low (< 0.05) risks releasing underfitted models affected by sampling noise that offer no practical business value. Setting the margin excessively high (> 0.90) rejects robust, production-ready models that offer substantial error reduction.

---

## CI/CD Pipeline Failure & Recovery Demonstration

Three sequential workflow runs demonstrate automated pipeline enforcement:

1. **Failure A — Model Quality Gate Breach:**
   * **Run Link:** https://github.com/prachig09/california-housing-mlops/actions/runs/35986229225/job/107589421589
   * **Cause:** Trained an underfit 1-tree `RandomForestRegressor` (`n_estimators=1`, `max_depth=1`).
   * **Outcome:** Quality gate check intercepted performance regression (`candidate_mse` of ~1.0021 failed the $\le 0.8762$ required threshold). The step exited via `sys.exit(1)` and blocked package publication.

2. **Failure B — Application Test Breach:**
   * **Run Link:** https://github.com/prachig09/california-housing-mlops/actions/runs/35986738469/job/107591073468
   * **Cause:** Injected schema test assertion error in `tests/test_app.py` by replacing `.drop(columns=["MedInc"])` with `.copy()`.
   * **Outcome:** `pytest` caught the broken contract test (DID NOT RAISE <class 'ValueError'>) and terminated execution with exit code 1, preventing deployment of a broken prediction interface.

3. **Final Successful Pipeline:**
   * **Run Link:** https://github.com/prachig09/california-housing-mlops/actions/runs/35987878283
   * **Artifact Package:** `california-housing-model-run-4` (Contains model.joblib, metrics_report.json, predict.py, requirements.txt).
---

## Questions
1. Why does your evaluation metric suit your task?

    MSE penalizes large estimation errors quadratically. In real estate pricing, underestimating or overestimating property values by $100,000+ carries severe financial risks compared to small $5,000 deviations, making MSE the ideal metric.

2. Why did you choose this improvement margin? What would happen if it were too low or too high?

    A margin of 0.50 requires a 36%+ reduction in MSE over guessing the median. A margin too low permits noisy, unhelpful models into deployment, while a margin too high rejects viable models that provide significant real-world improvements.

3. What caused each failed run? Which check prevented publication?

* `Failure A:` Severe underfitting (max_depth=1) failed the Train Baseline & Candidate Model quality gate check (sys.exit(1)).           
* `Failure B:` An invalid test payload failed the Run Application Tests step (python -m pytest tests/).

4. Which parts of your workflow demonstrate continuous integration and artifact delivery?

* `Continuous Integration (CI):` GitHub Actions automatically checking out code, validating data schema, training models, and executing pytest unit tests on every git push.
* `Artifact Delivery (CD):` The Publish Validated Model Package Artifact step executing conditionally (if: success()) to bundle and store the deployment archive.

5. Which MLOps maturity level best describes your implementation?

* `MLOps Level 1 (Automated Pipeline Execution).` The pipeline features script-driven automated training, validation, testing, and conditional artifact packaging. 
* `Reaching MLOps Level 2` would require live data/concept drift monitoring (e.g., Evidently AI), a centralized feature store (e.g., Feast), and continuous microservice endpoint deployment (e.g., Docker/Kubernetes).
