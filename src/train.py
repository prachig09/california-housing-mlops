import sys
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.data import load_and_validate, EXPECTED_FEATURES, TARGET_COL

# Quality Gate Configuration
# Metric: Mean Squared Error (MSE) on median house value ($100k units)
# Gate Condition: candidate_mse <= baseline_mse - MARGIN
MARGIN = 0.50  # Requires at least a $50k^2 reduction in MSE relative to the dummy baseline

def train_and_evaluate(artificially_fail_gate=False):
    df = load_and_validate()
    
    X = df[EXPECTED_FEATURES]
    y = df[TARGET_COL]
    
    # Reproducible split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 1. Baseline Model (Predicting Median)
    baseline_model = DummyRegressor(strategy="median")
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_val)
    baseline_mse = mean_squared_error(y_val, baseline_preds)
    
    # 2. Candidate Model Pipeline
    if artificially_fail_gate:
        # Failure A Trigger: Underfitted model with single decision tree layer
        candidate_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=1, max_depth=1, random_state=42))
        ])
    else:
        candidate_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42))
        ])
        
    # Fit preprocessing using ONLY training set
    candidate_pipeline.fit(X_train, y_train)
    candidate_preds = candidate_pipeline.predict(X_val)
    candidate_mse = mean_squared_error(y_val, candidate_preds)
    
    required_max_mse = baseline_mse - MARGIN
    gate_passed = bool(candidate_mse <= required_max_mse)
    
    metrics_report = {
        "dataset": "California Housing",
        "metric": "Mean Squared Error (MSE)",
        "baseline_score": round(float(baseline_mse), 4),
        "candidate_score": round(float(candidate_mse), 4),
        "margin": MARGIN,
        "required_max_score": round(float(required_max_mse), 4),
        "gate_passed": gate_passed
    }
    
    os.makedirs("artifact_build", exist_ok=True)
    with open("artifact_build/metrics_report.json", "w") as f:
        json.dump(metrics_report, f, indent=4)
        
    print("----- MODEL EVALUATION REPORT -----")
    print(json.dumps(metrics_report, indent=2))
    
    if not gate_passed:
        print(
            f"QUALITY GATE FAILED: Candidate MSE ({candidate_mse:.4f}) did not meet "
            f"required quality score (<= {required_max_mse:.4f}).", 
            file=sys.stderr
        )
        sys.exit(1)
        
    print("Quality gate passed! Saving inference artifact...")
    joblib.dump(candidate_pipeline, "artifact_build/model.joblib")

if __name__ == "__main__":
    fail_flag = "--fail-gate" in sys.argv
    train_and_evaluate(artificially_fail_gate=fail_flag)