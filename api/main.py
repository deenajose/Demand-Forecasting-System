from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

train_df = pd.read_csv(DATA_PATH)

train_df["date"] = pd.to_datetime(train_df["date"])

train_df = train_df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Demand Forecasting API",
    description="Machine learning API for store-item demand forecasting",
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ForecastRequest(BaseModel):

    store: int
    item: int
    days: int = 7


# ============================================================
# FEATURE CREATION
# ============================================================

def create_features(df):

    data = df.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    grouped = data.groupby(
        ["store", "item"]
    )["sales"]

    data["lag_1"] = grouped.transform(
        lambda x: x.shift(1)
    )

    data["lag_7"] = grouped.transform(
        lambda x: x.shift(7)
    )

    data["lag_14"] = grouped.transform(
        lambda x: x.shift(14)
    )

    data["lag_28"] = grouped.transform(
        lambda x: x.shift(28)
    )

    data["rolling_mean_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    data["rolling_mean_14"] = grouped.transform(
        lambda x: x.shift(1).rolling(14).mean()
    )

    data["rolling_mean_28"] = grouped.transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    data["rolling_std_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    return data


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
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Demand Forecasting API is running",
        "model": "XGBoost",
        "status": "online"
    }


# ============================================================
# STORES ENDPOINT
# ============================================================

@app.get("/stores")
def get_stores():

    stores = sorted(
        train_df["store"].unique().tolist()
    )

    return {
        "stores": stores
    }


# ============================================================
# ITEMS ENDPOINT
# ============================================================

@app.get("/items")
def get_items():

    items = sorted(
        train_df["item"].unique().tolist()
    )

    return {
        "items": items
    }


# ============================================================
# HISTORICAL DATA ENDPOINT
# ============================================================

@app.get("/history/{store}/{item}")
def get_history(
    store: int,
    item: int,
    days: int = 30
):

    if store not in train_df["store"].unique():

        raise HTTPException(
            status_code=404,
            detail="Store not found"
        )

    if item not in train_df["item"].unique():

        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )

    if days < 1 or days > 365:

        raise HTTPException(
            status_code=400,
            detail="Days must be between 1 and 365"
        )

    history = train_df[
        (train_df["store"] == store) &
        (train_df["item"] == item)
    ].sort_values("date").tail(days)

    return {
        "store": store,
        "item": item,
        "history_days": len(history),
        "history": [
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "sales": float(row["sales"])
            }

            for _, row in history.iterrows()
        ]
    }


# ============================================================
# FORECAST ENDPOINT
# ============================================================

@app.post("/forecast")
def forecast_demand(request: ForecastRequest):

    store = request.store
    item = request.item
    days = request.days

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if store not in train_df["store"].unique():

        raise HTTPException(
            status_code=404,
            detail="Store not found"
        )

    if item not in train_df["item"].unique():

        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )

    if days < 1 or days > 30:

        raise HTTPException(
            status_code=400,
            detail="Forecast days must be between 1 and 30"
        )

    # --------------------------------------------------------
    # SELECT STORE + ITEM
    # --------------------------------------------------------

    history = train_df[
        (train_df["store"] == store) &
        (train_df["item"] == item)
    ].copy()

    history = history.sort_values("date")

    if len(history) < 30:

        raise HTTPException(
            status_code=400,
            detail="Not enough historical data"
        )

    # --------------------------------------------------------
    # WORKING DATASET
    # --------------------------------------------------------

    working_df = history[
        ["date", "store", "item", "sales"]
    ].copy()

    forecasts = []

    # --------------------------------------------------------
    # ITERATIVE FORECASTING
    # --------------------------------------------------------

    for _ in range(days):

        next_date = (
            working_df["date"].max()
            + pd.Timedelta(days=1)
        )

        new_row = pd.DataFrame(
            {
                "date": [next_date],
                "store": [store],
                "item": [item],
                "sales": [np.nan]
            }
        )

        working_df = pd.concat(
            [working_df, new_row],
            ignore_index=True
        )

        featured_df = create_features(
            working_df
        )

        current_row = featured_df.iloc[[-1]].copy()

        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        if current_row[FEATURE_COLUMNS].isnull().any().any():

            raise HTTPException(
                status_code=500,
                detail="Unable to create forecasting features"
            )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            current_row[FEATURE_COLUMNS]
        )[0]

        prediction = max(
            0,
            float(prediction)
        )

        # ----------------------------------------------------
        # SAVE PREDICTION
        # ----------------------------------------------------

        working_df.loc[
            working_df.index[-1],
            "sales"
        ] = prediction

        forecasts.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_demand": round(
                    prediction,
                    2
                )
            }
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    predictions = [
        x["predicted_demand"]
        for x in forecasts
    ]

    total_demand = sum(predictions)

    average_demand = (
        total_demand / len(predictions)
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "store": store,
        "item": item,
        "forecast_days": days,
        "total_predicted_demand": round(
            total_demand,
            2
        ),
        "average_daily_demand": round(
            average_demand,
            2
        ),
        "forecast": forecasts
    }