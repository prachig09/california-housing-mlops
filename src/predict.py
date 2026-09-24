import os
import joblib
import pandas as pd
from src.data import EXPECTED_FEATURES

def load_model(model_path="artifact_build/model.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def predict(input_data: pd.DataFrame, model_path="artifact_build/model.joblib"):
    model = load_model(model_path)
    
    missing = [c for c in EXPECTED_FEATURES if c not in input_data.columns]
    if missing:
        raise ValueError(f"Input dataframe missing required features: {missing}")
        
    return model.predict(input_data)

if __name__ == "__main__":
    sample_data = pd.DataFrame([{
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }])
    preds = predict(sample_data)
    print(f"Predicted Median House Value: ${preds[0] * 100000:,.2f}")