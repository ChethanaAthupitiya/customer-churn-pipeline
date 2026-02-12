import pandas as pd
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    
    raw_path = "data/raw/churn.csv"
    processed_path = "data/processed/data.csv"

    df = load_data(raw_path)

    os.makedirs("data/processed", exist_ok=True)

    save_data(df, processed_path)

    print("Data ingestion completed.")
