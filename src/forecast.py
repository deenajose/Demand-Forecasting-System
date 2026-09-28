import pandas as pd
import numpy as np
import joblib

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "train.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "xgboost_demand_model.pkl"
)

OUTPUT_PATH = (
    BASE_DIR
    / "outputs"
    / "future_forecast.csv"
)


FEATURES = [
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


# --------------------------------------------------
# FEATURE CREATION FOR FORECASTING
# --------------------------------------------------

def create_features(df):

    df = df.copy()

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

    grouped = df.groupby(
        ["store", "item"]
    )["sales"]

    df["lag_1"] = grouped.shift(1)

    df["lag_7"] = grouped.shift(7)

    df["lag_14"] = grouped.shift(14)

    df["lag_28"] = grouped.shift(28)

    df["rolling_mean_7"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .mean()
        )
    )

    df["rolling_mean_14"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(14)
            .mean()
        )
    )

    df["rolling_mean_28"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(28)
            .mean()
        )
    )

    df["rolling_std_7"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .std()
        )
    )

    return df


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("=" * 60)
print("FUTURE DEMAND FORECASTING")
print("=" * 60)

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    TRAIN_PATH
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)


# --------------------------------------------------
# SELECT STORE + ITEM
# --------------------------------------------------

STORE_ID = 1

ITEM_ID = 1

FORECAST_DAYS = 7


history = df[
    (df["store"] == STORE_ID)
    &
    (df["item"] == ITEM_ID)
].copy()


print(
    f"\nForecasting Store {STORE_ID}, "
    f"Item {ITEM_ID}"
)

print(
    f"Historical data: "
    f"{history['date'].min().date()} "
    f"→ "
    f"{history['date'].max().date()}"
)


# --------------------------------------------------
# ITERATIVE FORECASTING
# --------------------------------------------------

forecasts = []

working_data = history.copy()


for _ in range(FORECAST_DAYS):

    next_date = (
        working_data["date"].max()
        + pd.Timedelta(days=1)
    )

    # Add placeholder row
    new_row = pd.DataFrame(
        {
            "date": [next_date],
            "store": [STORE_ID],
            "item": [ITEM_ID],
            "sales": [np.nan]
        }
    )

    working_data = pd.concat(
        [
            working_data,
            new_row
        ],
        ignore_index=True
    )

    # Create features
    featured = create_features(
        working_data
    )

    current = featured[
        featured["date"] == next_date
    ].copy()

    X = current[FEATURES]

    prediction = model.predict(X)[0]

    prediction = max(
        0,
        prediction
    )

    # Store forecast
    forecasts.append(
        {
            "date": next_date,
            "store": STORE_ID,
            "item": ITEM_ID,
            "predicted_sales": prediction
        }
    )

    # Feed prediction back into history
    working_data.loc[
        working_data["date"] == next_date,
        "sales"
    ] = prediction


# --------------------------------------------------
# SAVE FORECAST
# --------------------------------------------------

forecast_df = pd.DataFrame(
    forecasts
)

forecast_df[
    "predicted_sales"
] = forecast_df[
    "predicted_sales"
].round(2)


forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\n" + "=" * 60)

print("7-DAY DEMAND FORECAST")

print("=" * 60)

print(
    forecast_df.to_string(
        index=False
    )
)

print(
    "\nTotal predicted demand:",
    round(
        forecast_df[
            "predicted_sales"
        ].sum(),
        2
    )
)

print("\nForecast saved to:")

print(OUTPUT_PATH)

print("\nForecast completed successfully!")