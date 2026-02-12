import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv("data/raw/churn.csv")
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = load_data()

    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = joblib.load("artifacts/preprocessor.joblib")

    model = RandomForestClassifier(n_estimators=300, random_state=42)

    churn_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    churn_pipeline.fit(X_train, y_train)

    preds = churn_pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(churn_pipeline, "artifacts/churn_pipeline.joblib")

    print(f"✅ Pipeline trained & saved: artifacts/churn_pipeline.joblib | Accuracy: {acc:.4f}")
