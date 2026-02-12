# src/transformation.py
from sklearn.preprocessing import StandardScaler

def transform_data(df):
    """
    Separate features (X) and target (y)
    Scale features using StandardScaler
    """
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.ingestion import load_data
    from src.validation import validate_data

    df = load_data()
    df = validate_data(df)
    X_scaled, y, scaler = transform_data(df)
    print("Transformed features shape:", X_scaled.shape)
