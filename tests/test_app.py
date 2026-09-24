import pytest
import pandas as pd
import numpy as np
import os
from src.predict import load_model, predict

MODEL_PATH = "artifact_build/model.joblib"

@pytest.fixture
def valid_sample():
    return pd.DataFrame([{
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }])

def test_model_artifact_exists_and_loads():
    """Verify that trained pipeline artifact exists and can be deserialized."""
    assert os.path.exists(MODEL_PATH), "Model artifact was not produced."
    model = load_model(MODEL_PATH)
    assert model is not None

def test_prediction_output_shape_and_type(valid_sample):
    """Verify inference output format and dimensionality."""
    predictions = predict(valid_sample, model_path=MODEL_PATH)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 1
    assert isinstance(predictions[0], (float, np.floating))

def test_missing_feature_rejection(valid_sample):
    """Verify schema mismatch rejection with clear error message."""
    invalid_sample = valid_sample.drop(columns=["MedInc"])
    
    with pytest.raises(ValueError, match="Input dataframe missing required features"):
        predict(invalid_sample, model_path=MODEL_PATH)