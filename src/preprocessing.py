import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    target = "Churn"
    X = df.drop(columns=[target])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols)
        ]
    )

    return preprocessor, num_cols, cat_cols


if __name__ == "__main__":
    df = load_data("data/raw/churn.csv")
    df = basic_clean(df)

    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, "artifacts/columns.joblib")

    print("✅ Preprocessor saved to artifacts/preprocessor.joblib")
