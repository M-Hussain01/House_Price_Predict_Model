# src/model.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(X, y):
    """
    Train Linear Regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    """
    Evaluate model using RMSE
    """
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)  # squared=False ki jagah manually sqrt
    print(f"RMSE: {rmse:.3f}")
    return rmse

# Optional test run
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.ingestion import load_data
    from src.validation import validate_data
    from src.transformation import transform_data

    # Load & validate
    df = load_data()
    df = validate_data(df)

    # Transform
    X_scaled, y, _ = transform_data(df)

    # Train model
    model = train_model(X_scaled, y)

    # Evaluate
    evaluate_model(model, X_scaled, y)
