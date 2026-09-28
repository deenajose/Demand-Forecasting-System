import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "featured_data.csv"


def create_features(df):
    """
    Create time-series and historical demand features.
    """

    df = df.copy()

    # --------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------
    # GROUP BY STORE + ITEM
    # --------------------------------------------------

    grouped = df.groupby(
        ["store", "item"],
        group_keys=False
    )["sales"]

    # --------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)
    df["lag_28"] = grouped.shift(28)

    # --------------------------------------------------
    # ROLLING FEATURES
    #
    # IMPORTANT:
    # Shift first so today's sales are NOT used
    # to predict today's sales.
    # --------------------------------------------------

    df["rolling_mean_7"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["rolling_mean_14"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x: x.shift(1).rolling(14).mean()
        )
    )

    df["rolling_mean_28"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x: x.shift(1).rolling(28).mean()
        )
    )

    # --------------------------------------------------
    # ROLLING STANDARD DEVIATION
    # --------------------------------------------------

    df["rolling_std_7"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x: x.shift(1).rolling(7).std()
        )
    )

    # --------------------------------------------------
    # REMOVE ROWS WITH MISSING VALUES
    #
    # First 28 days of every store/item combination
    # don't have enough historical data for all features.
    # --------------------------------------------------

    df = df.dropna().reset_index(drop=True)

    return df


def main():

    print("=" * 60)
    print("DEMAND FORECASTING - FEATURE ENGINEERING")
    print("=" * 60)

    print("\nLoading dataset...")

    df = pd.read_csv(TRAIN_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    print(f"Original rows: {len(df):,}")

    print("\nCreating features...")

    df = create_features(df)

    print(
        f"Rows after feature engineering: "
        f"{len(df):,}"
    )

    print("\nFeatures created:")

    for column in df.columns:
        print(f"  - {column}")

    print("\nSaving processed dataset...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\nFeature engineering completed successfully!")


if __name__ == "__main__":
    main()