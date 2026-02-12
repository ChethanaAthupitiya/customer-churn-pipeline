import pandas as pd
import joblib
import os

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_test_data():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return acc, report, cm


def save_results(model_name, acc, report, cm):
    os.makedirs("reports", exist_ok=True)

    with open("reports/results.txt", "a", encoding="utf-8") as f:
        f.write(f"\n--- {model_name} ---\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")


if __name__ == "__main__":

    X_test, y_test = load_test_data()

    model_files = [
        "models/logistic_regression.joblib",
        "models/random_forest.joblib"
    ]

    best_model = None
    best_acc = 0
    best_name = ""

    for file in model_files:
        model = joblib.load(file)
        name = os.path.basename(file)

        acc, report, cm = evaluate_model(model, X_test, y_test)

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("Confusion Matrix:\n", cm)

        save_results(name, acc, report, cm)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # Save best model separately
    joblib.dump(best_model, "models/best_model.joblib")

    print(f"\n✅ Best model saved as models/best_model.joblib ({best_name})")
