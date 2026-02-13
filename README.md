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

```
customer-churn-pipeline/
│
├── app/
│   └── main.py              # FastAPI application
│
├── src/
│   ├── preprocessing.py     # Data preprocessing
│   ├── training.py          # Model training
│   ├── train_pipeline.py    # Pipeline training
│   └── evaluation.py        # Model evaluation
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
---

## 📊 Model Performance

- Model: Random Forest Classifier
- Accuracy: **78.96%**
- Pipeline includes preprocessing + model

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ChethanaAthupitiya/customer-churn-pipeline.git
cd customer-churn-pipeline

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏃 Run the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Open your browser and go to:

```
http://127.0.0.1:8000/docs
```

You will see the Swagger UI where you can test predictions.

---

## 🧪 Example Prediction Request

POST `/predict`

```json
{
  "data": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.7,
    "TotalCharges": 95.7
  }
}
```

Example Response:

```json
{
  "prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.85
}
```

---

## 🎯 Use Cases

- Telecom customer churn prediction
- Machine Learning pipeline demonstration
- FastAPI deployment example
- Data Science portfolio project

---

## 👩‍💻 Author

Chethana Athupitiya  
Electrical & Electronic Engineering Graduate  
Interested in Machine Learning & Data Science  

GitHub: https://github.com/ChethanaAthupitiya

---

## ⭐ Future Improvements

- Add Docker support
- Deploy to cloud (AWS / Render)
- Add model monitoring