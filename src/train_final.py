import pandas as pd
import joblib
from pathlib import Path
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    "final_xgboost_demand_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading historical data...")

df = pd.read_csv(
    DATA_PATH
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)


print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("Creating forecasting features...")


df["year"] = df["date"].dt.year

df["month"] = df["date"].dt.month

df["day"] = df["date"].dt.day

df["day_of_week"] = (
    df["date"].dt.dayofweek
)

df["week_of_year"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# GROUPED SALES
# ============================================================

grouped = df.groupby(
    ["store", "item"]
)["sales"]


# ============================================================
# LAG FEATURES
# ============================================================

df["lag_1"] = grouped.transform(
    lambda x: x.shift(1)
)

df["lag_7"] = grouped.transform(
    lambda x: x.shift(7)
)

df["lag_14"] = grouped.transform(
    lambda x: x.shift(14)
)

df["lag_28"] = grouped.transform(
    lambda x: x.shift(28)
)


# ============================================================
# ROLLING FEATURES
# ============================================================

df["rolling_mean_7"] = grouped.transform(
    lambda x:
    x.shift(1)
    .rolling(7)
    .mean()
)

df["rolling_mean_14"] = grouped.transform(
    lambda x:
    x.shift(1)
    .rolling(14)
    .mean()
)

df["rolling_mean_28"] = grouped.transform(
    lambda x:
    x.shift(1)
    .rolling(28)
    .mean()
)

df["rolling_std_7"] = grouped.transform(
    lambda x:
    x.shift(1)
    .rolling(7)
    .std()
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [

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


# ============================================================
# REMOVE INITIAL NULL ROWS
# ============================================================

model_df = df.dropna(
    subset=FEATURE_COLUMNS
).copy()


print(
    f"Training rows: {len(model_df):,}"
)


# ============================================================
# X AND Y
# ============================================================

X = model_df[
    FEATURE_COLUMNS
]

y = model_df[
    "sales"
]


# ============================================================
# FINAL XGBOOST MODEL
# ============================================================

print(
    "\nTraining final XGBoost model..."
)


model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=8,

    min_child_weight=3,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


print(
    "\nFinal model saved successfully:"
)

print(
    MODEL_PATH
)

print(
    "\nFinal production model training complete."
)