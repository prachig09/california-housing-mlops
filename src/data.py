import sys
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

EXPECTED_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]
TARGET_COL = "MedHouseVal"

def fetch_and_save_data():
    """Downloads California Housing dataset and saves locally."""
    os.makedirs("data", exist_ok=True)
    data_path = os.path.join("data", "california_housing.csv")
    
    if not os.path.exists(data_path):
        print("Fetching California Housing dataset...")
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df.rename(columns={"MedHouseVal": TARGET_COL}, inplace=True)
        df.to_csv(data_path, index=False)
        print("Dataset successfully saved.")

def validate_schema(df: pd.DataFrame):
    """Validates required features and target columns in the DataFrame."""
    missing_features = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_features:
        print(f"SCHEMA ERROR: Missing feature columns: {missing_features}", file=sys.stderr)
        sys.exit(1)
        
    if TARGET_COL not in df.columns:
        print(f"SCHEMA ERROR: Missing target column: '{TARGET_COL}'", file=sys.stderr)
        sys.exit(1)
        
    print("Schema validation passed.")

def load_and_validate():
    fetch_and_save_data()
    data_path = os.path.join("data", "california_housing.csv")
    df = pd.read_csv(data_path)
    validate_schema(df)
    return df

if __name__ == "__main__":
    load_and_validate()