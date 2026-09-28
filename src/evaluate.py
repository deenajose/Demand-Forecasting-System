import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    BASE_DIR /
    "outputs" /
    "test_predictions.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("Loading test predictions...")

df = pd.read_csv(
    PREDICTIONS_FILE
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# FIND ACTUAL + PREDICTED COLUMNS
# ============================================================

actual_column = None
prediction_column = None

possible_actual = [
    "sales",
    "actual",
    "actual_sales",
    "y_true"
]

possible_prediction = [
    "prediction",
    "predicted",
    "predicted_sales",
    "y_pred"
]


for column in possible_actual:

    if column in df.columns:

        actual_column = column
        break


for column in possible_prediction:

    if column in df.columns:

        prediction_column = column
        break


if actual_column is None:

    raise ValueError(
        "Could not find actual sales column."
    )


if prediction_column is None:

    raise ValueError(
        "Could not find prediction column."
    )


print(
    f"\nActual column: {actual_column}"
)

print(
    f"Prediction column: {prediction_column}"
)


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

df = df[
    [
        actual_column,
        prediction_column
    ]
].dropna()


y_true = df[actual_column]

y_pred = df[prediction_column]


# ============================================================
# MAE
# ============================================================

mae = mean_absolute_error(
    y_true,
    y_pred
)


# ============================================================
# RMSE
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)


# ============================================================
# MAPE
# ============================================================

non_zero_mask = y_true != 0

mape = np.mean(
    np.abs(
        (
            y_true[non_zero_mask]
            -
            y_pred[non_zero_mask]
        )
        /
        y_true[non_zero_mask]
    )
) * 100


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 50)

print("MODEL PERFORMANCE")

print("=" * 50)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"MAPE : {mape:.2f}%"
)

print("=" * 50)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "MAE": float(mae),

    "RMSE": float(rmse),

    "MAPE": float(mape)

}


metrics_file = (
    OUTPUT_DIR /
    "model_metrics.json"
)


with open(
    metrics_file,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    f"\nMetrics saved to: {metrics_file}"
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plot_df = pd.DataFrame(
    {
        "Actual": y_true.values,
        "Predicted": y_pred.values
    }
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    plot_df["Actual"],
    label="Actual"
)

plt.plot(
    plot_df["Predicted"],
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Demand"
)

plt.xlabel(
    "Test Observations"
)

plt.ylabel(
    "Demand"
)

plt.legend()

plt.tight_layout()


plot_path = (
    OUTPUT_DIR /
    "actual_vs_predicted.png"
)


plt.savefig(
    plot_path,
    dpi=150
)

plt.close()


print(
    f"Plot saved to: {plot_path}"
)


# ============================================================
# RECENT ACTUAL VS PREDICTED
# ============================================================

recent_df = plot_df.tail(200)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    recent_df["Actual"].values,
    label="Actual"
)

plt.plot(
    recent_df["Predicted"].values,
    label="Predicted"
)

plt.title(
    "Recent Actual vs Predicted Demand"
)

plt.xlabel(
    "Recent Test Observations"
)

plt.ylabel(
    "Demand"
)

plt.legend()

plt.tight_layout()


recent_plot_path = (
    OUTPUT_DIR /
    "recent_actual_vs_predicted.png"
)


plt.savefig(
    recent_plot_path,
    dpi=150
)

plt.close()


print(
    f"Recent plot saved to: {recent_plot_path}"
)


print("\nEvaluation complete.")