import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"


def load_train_data():
    df = pd.read_csv(TRAIN_PATH)
    return df


def load_test_data():
    df = pd.read_csv(TEST_PATH)
    return df


def prepare_data(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        by=["store", "item", "date"]
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    train = load_train_data()
    train = prepare_data(train)

    print("\nDataset loaded successfully!")
    print("-" * 50)

    print(f"Rows: {len(train):,}")
    print(f"Columns: {len(train.columns)}")

    print("\nColumns:")
    print(train.columns.tolist())

    print("\nFirst 5 rows:")
    print(train.head())

    print("\nData types:")
    print(train.dtypes)

    print("\nMissing values:")
    print(train.isnull().sum())

    print("\nDate range:")
    print(train["date"].min(), "to", train["date"].max())

    print("\nNumber of stores:")
    print(train["store"].nunique())

    print("\nNumber of items:")
    print(train["item"].nunique())