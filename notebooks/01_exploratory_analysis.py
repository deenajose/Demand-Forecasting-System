import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date")


print("=" * 60)
print("DEMAND FORECASTING - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nNumber of stores:", df["store"].nunique())
print("Number of products:", df["item"].nunique())


# --------------------------------------------------
# DAILY TOTAL SALES
# --------------------------------------------------

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["sales"]
)

plt.title("Daily Total Demand")
plt.xlabel("Date")
plt.ylabel("Units Sold")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "daily_demand.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# MONTHLY DEMAND
# --------------------------------------------------

monthly_sales = (
    df.set_index("date")
    .resample("ME")["sales"]
    .sum()
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values
)

plt.title("Monthly Total Demand")
plt.xlabel("Month")
plt.ylabel("Units Sold")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_demand.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# STORE DEMAND
# --------------------------------------------------

store_sales = (
    df.groupby("store")["sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=store_sales.index,
    y=store_sales.values
)

plt.title("Total Sales by Store")
plt.xlabel("Store")
plt.ylabel("Units Sold")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "store_demand.png",
    dpi=150
)

plt.show()


# --------------------------------------------------
# PRODUCT DEMAND
# --------------------------------------------------

item_sales = (
    df.groupby("item")["sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=item_sales.index,
    y=item_sales.values
)

plt.title("Total Sales by Product")
plt.xlabel("Product")
plt.ylabel("Units Sold")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "product_demand.png",
    dpi=150
)

plt.show()


print("\nEDA completed successfully.")

print("\nGenerated files:")

for file in OUTPUT_DIR.glob("*.png"):
    print(file.name)