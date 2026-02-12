import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_data(df):
    """
    Validate dataset: check missing values
    Fill missing values if any
    """
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Filling missing values...")
        df = df.fillna(method='ffill')
    else:
        print("No missing values found.")
    return df


if __name__ == "__main__":
    import pandas as pd
    from src.ingestion import load_data

    df = load_data()
    df = validate_data(df)
    print(df.isnull().sum())
