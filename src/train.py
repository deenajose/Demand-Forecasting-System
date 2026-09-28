import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "featured_data.csv"
)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("XGBOOST DEMAND FORECASTING MODEL")
print("=" * 60)

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows: {len(df):,}")


# --------------------------------------------------
# TIME-BASED SPLIT
# --------------------------------------------------

split_date = df["date"].quantile(0.8)

train = df[
    df["date"] < split_date
].copy()

test = df[
    df["date"] >= split_date
].copy()


print("\nTime-based split")

print(
    f"Training: {train['date'].min().date()} "
    f"→ {train['date'].max().date()}"
)

print(
    f"Testing : {test['date'].min().date()} "
    f"→ {test['date'].max().date()}"
)


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "store",
    "item",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7"
]


X_train = train[features]

y_train = train["sales"]

X_test = test[features]

y_test = test["sales"]


print("\nNumber of features:", len(features))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(X_test)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


# MAPE with protection against zero values
non_zero = y_test != 0

mape = (
    np.mean(
        np.abs(
            (
                y_test[non_zero]
                - predictions[non_zero]
            )
            / y_test[non_zero]
        )
    )
    * 100
)


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nMAE  : {mae:.2f}")

print(f"RMSE : {rmse:.2f}")

print(f"MAPE : {mape:.2f}%")


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_path = (
    MODEL_DIR
    / "xgboost_demand_model.pkl"
)

joblib.dump(
    model,
    model_path
)


print("\nModel saved to:")

print(model_path)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame(
    {
        "feature": features,
        "importance": model.feature_importances_
    }
)

importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\nTop feature importance:")

print(
    importance.head(10).to_string(
        index=False
    )
)


# --------------------------------------------------
# SAVE TEST PREDICTIONS
# --------------------------------------------------

results = test[
    ["date", "store", "item", "sales"]
].copy()

results["predicted_sales"] = predictions

results_path = (
    BASE_DIR
    / "outputs"
    / "test_predictions.csv"
)

results.to_csv(
    results_path,
    index=False
)


print("\nPredictions saved to:")

print(results_path)

print("\nTraining completed successfully!")