# Customer Churn Prediction Pipeline with FastAPI

An end-to-end Machine Learning pipeline to predict customer churn using the Telco Customer Churn dataset.  
This project includes data preprocessing, model training, evaluation, and deployment as a REST API using FastAPI.

---

## 🚀 Features

- End-to-end ML pipeline (data → preprocessing → training → deployment)
- Preprocessing using scikit-learn (OneHotEncoder, imputation)
- Random Forest model with **78.96% accuracy**
- FastAPI REST API for real-time predictions
- Swagger UI for interactive testing
- Production-ready pipeline artifact

---

## 🧠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Git & GitHub

---

## 📂 Project Structure
customer-churn-pipeline/
│
├── app/
│   └── main.py              # FastAPI application
│
├── src/
│   ├── preprocessing.py     # Data preprocessing
│   ├── training.py         # Model training
│   ├── train_pipeline.py   # Pipeline training
│   └── evaluation.py       # Model evaluation
│
├── artifacts/
│   ├── churn_pipeline.joblib
│   ├── preprocessor.joblib
│   └── columns.joblib
│
├── data/
├── models/
├── reports/
├── notebooks/
│
└── README.md
```







