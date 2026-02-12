import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    return X_train, y_train


def train_models(X_train, y_train):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def save_models(trained_models):
    os.makedirs("models", exist_ok=True)

    for name, model in trained_models.items():
        joblib.dump(model, f"models/{name}.joblib")


if __name__ == "__main__":
    X_train, y_train = load_processed_data()

    trained_models = train_models(X_train, y_train)

    save_models(trained_models)

    print("Training completed. Models saved in /models folder.")
