# 📈 AI Demand Forecasting System

An end-to-end machine learning application that predicts future product demand for individual stores using historical sales data.

The project combines **time-series feature engineering, XGBoost regression, model evaluation, FastAPI and Streamlit** into a complete production-style forecasting workflow.

---

## 🚀 Project Overview

Retail businesses need accurate demand estimates to support inventory planning, purchasing and operational decisions.

This project takes historical store-product sales data and predicts future demand for a selected:

- Store
- Product
- Forecast horizon

The system provides both a machine learning backend and an interactive dashboard.

---

## 🧠 What the System Does

```text
Historical Sales Data
        ↓
Data Preprocessing
        ↓
Time-Series Feature Engineering
        ↓
Lag Features + Rolling Statistics
        ↓
XGBoost Regression
        ↓
Model Evaluation
        ↓
Final Production Model
        ↓
FastAPI
        ↓
Streamlit Dashboard
        ↓
Future Demand Forecast
```

````

---

## ✨ Features

### Machine Learning

- XGBoost regression model
- Time-series aware train/test evaluation
- Lag features
- Rolling mean features
- Rolling standard deviation
- Calendar features
- Store and product information

### Forecasting

- 1–30 day forecasting horizon
- Store-specific forecasts
- Product-specific forecasts
- Iterative multi-step forecasting

### API

FastAPI backend provides:

```text
GET  /
GET  /stores
GET  /items
GET  /history/{store}/{item}
POST /forecast
```

### Dashboard

Streamlit dashboard provides:

- Store selection
- Product selection
- Forecast horizon selection
- Total predicted demand
- Average daily demand
- Peak demand
- Lowest demand
- Historical demand visualization
- Future forecast visualization
- Detailed forecast table
- CSV download
- Model performance metrics

---

# 📊 Dataset

The project uses the **Store Item Demand Forecasting Challenge** dataset.

The training data contains:

- `date`
- `store`
- `item`
- `sales`

The dataset contains historical daily sales for multiple stores and products.

The raw dataset is intentionally excluded from GitHub through `.gitignore`.

---

# 🛠️ Technologies

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming          |
| Pandas       | Data processing           |
| NumPy        | Numerical computation     |
| Scikit-learn | Evaluation                |
| XGBoost      | Forecasting model         |
| Matplotlib   | Evaluation visualization  |
| Plotly       | Interactive visualization |
| FastAPI      | REST API                  |
| Streamlit    | Dashboard                 |
| Joblib       | Model serialization       |

---

# 🔧 Feature Engineering

The model uses several categories of features.

## Calendar Features

```text
year
month
day
day_of_week
week_of_year
is_weekend
```

## Lag Features

```text
lag_1
lag_7
lag_14
lag_28
```

These capture previous demand patterns.

## Rolling Features

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
rolling_std_7
```

These capture recent demand trends and variability.

---

# 🤖 Machine Learning Model

The primary forecasting model is:

**XGBoost Regressor**

The model predicts:

```text
Future Daily Sales
```

for a specific store-product combination.

The final production model is trained using all available historical training observations after the evaluation process.

---

# 📏 Model Evaluation

The forecasting system evaluates the model using:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted demand.

### RMSE

Root Mean Squared Error gives greater weight to larger prediction errors.

### MAPE

Mean Absolute Percentage Error expresses prediction error as a percentage.

The evaluation uses a temporal holdout period rather than randomly shuffling time-series observations.

---

# 🖥️ Running the Project

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Demand-Forecasting-System
```

---

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Add the dataset

Place:

```text
train.csv
test.csv
```

inside:

```text
data/raw/
```

---

# 🧪 Train the Model

The normal evaluation/training pipeline can be run using the project scripts.

Example:

```powershell
python -m src.train
```

---

# 📊 Evaluate the Model

```powershell
python -m src.evaluate
```

This generates model metrics and evaluation plots inside:

```text
outputs/
```

---

# 🚀 Train the Final Production Model

After evaluation:

```powershell
python -m src.train_final
```

This creates:

```text
models/final_xgboost_demand_model.pkl
```

---

# 🌐 Start the FastAPI Backend

```powershell
uvicorn api.main:app --reload --port 8000
```

API:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 📈 Start the Streamlit Dashboard

Open another terminal:

```powershell
streamlit run dashboard/app.py
```

Dashboard:

```text
http://localhost:8501
```

---

# 🔮 Example API Request

```json
{
  "store": 1,
  "item": 1,
  "days": 7
}
```

Example response structure:

```json
{
  "store": 1,
  "item": 1,
  "forecast_days": 7,
  "total_predicted_demand": 1234.56,
  "average_daily_demand": 176.37,
  "forecast": [
    {
      "date": "2018-01-01",
      "predicted_demand": 170.42
    }
  ]
}
```

---

# 🏗️ Project Structure

```text
Demand-Forecasting-System/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   └── 01_exploratory_analysis.py
│
├── outputs/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── train_final.py
│   ├── forecast.py
│   └── evaluate.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🔐 Data and Model Files

Large datasets and trained model binaries are excluded from version control.

The following directories may contain locally generated files:

```text
data/raw/
data/processed/
models/
outputs/
```

---

# 💼 Why This Project Matters

Demand forecasting is a practical machine learning problem with applications in:

- Retail
- E-commerce
- Supply chain
- Inventory planning
- Procurement
- Logistics

The project demonstrates more than model training. It covers the complete ML application lifecycle:

```text
Data
 ↓
EDA
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Evaluation
 ↓
Production Model
 ↓
API
 ↓
Dashboard
```

---

# 🔮 Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Additional forecasting algorithms
- Holiday/event features
- Promotion information
- External economic features
- Automated retraining
- Model monitoring
- Prediction intervals
- Cloud deployment
- Containerization with Docker
- Automated CI/CD

---

# 👩‍💻 Author

**Deena Jose**

AI / ML / Data Science Project

````
