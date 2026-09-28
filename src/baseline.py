import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"


df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)


# --------------------------------------------------
# CREATE LAG-1 FEATURE
# --------------------------------------------------

df["previous_day_sales"] = (
    df.groupby(["store", "item"])["sales"]
    .shift(1)
)


# Remove first observation of each series
df = df.dropna(
    subset=["previous_day_sales"]
)


# --------------------------------------------------
# TIME-BASED SPLIT
# --------------------------------------------------

split_date = df["date"].quantile(0.8)

train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]


print("=" * 60)
print("NAIVE DEMAND FORECASTING BASELINE")
print("=" * 60)

print("\nTraining period:")
print(train["date"].min(), "to", train["date"].max())

print("\nTesting period:")
print(test["date"].min(), "to", test["date"].max())


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_true = test["sales"]
y_pred = test["previous_day_sales"]


mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)


print("\nRESULTS")
print("-" * 40)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")