# src/ingestion.py
import os
import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_data():
    """
    Load California Housing dataset.
    Save CSV locally for Streamlit deploy.
    """
    folder_path = "data"
    file_path = os.path.join(folder_path, "california.csv")
    
    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Check if CSV exists
    if os.path.exists(file_path):
        print("File already exists. Skipping download.")
        return pd.read_csv(file_path)
    
    # Load dataset
    california = fetch_california_housing(as_frame=True)
    df = california.frame  # Already a DataFrame with target
    
    # Save CSV
    df.to_csv(file_path, index=False)
    print(f"Dataset downloaded and saved at {file_path}")
    
    return df

# Optional test
if __name__ == "__main__":
    df = load_data()
    print(df.head())
