"""
Walmart Demand Forecasting Project
Model training and evaluation.

Models:
1. Seasonal Naive Baseline
2. Random Forest Regressor

Forecasting scope:
Daily aggregate Walmart Foods category sales across all stores.

Important:
Lag and rolling features are created BEFORE rows with missing
feature values are removed.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "foods_daily_demand.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Create output directories if they do not already exist.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD THE PROCESSED DATASET
# ============================================================

print("=" * 65)
print("WALMART FOODS DEMAND FORECASTING")
print("=" * 65)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{DATA_FILE}\n\n"
        "Run src/prepare_forecasting_data.py first."
    )

df = pd.read_csv(DATA_FILE)

required_input_columns = ["date", "units_sold"]

missing_input_columns = [
    column
    for column in required_input_columns
    if column not in df.columns
]

if missing_input_columns:
    raise ValueError(
        f"The processed dataset is missing required columns: "
        f"{missing_input_columns}"
    )

df["date"] = pd.to_datetime(df["date"], errors="raise")
df["units_sold"] = pd.to_numeric(
    df["units_sold"],
    errors="raise"
)

df = df.sort_values("date").reset_index(drop=True)

print("\nDATASET INFORMATION")
print("-" * 65)
print(f"Input file: {DATA_FILE}")
print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")
print(f"First date: {df['date'].min().date()}")
print(f"Last date: {df['date'].max().date()}")
print(f"Missing dates: {df['date'].duplicated().sum():,} duplicate dates")


# ============================================================
# 3. VALIDATE THE TIME SERIES
# ============================================================

if df["date"].duplicated().any():
    raise ValueError(
        "Duplicate dates were found. Each date must have one "
        "aggregate sales observation."
    )

if df["units_sold"].isna().any():
    raise ValueError(
        "Missing values were found in units_sold."
    )

if (df["units_sold"] < 0).any():
    raise ValueError(
        "Negative sales values were found. Investigate these "
        "before training the forecasting models."
    )

if not df["date"].is_monotonic_increasing:
    raise ValueError(
        "Dates are not in chronological order."
    )

date_differences = df["date"].diff().dropna()

if not date_differences.eq(pd.Timedelta(days=1)).all():
    raise ValueError(
        "The dataset does not contain consecutive daily dates. "
        "Check the data preparation step."
    )

print("\nTIME SERIES VALIDATION")
print("-" * 65)
print("Dates are unique: Yes")
print("Dates are consecutive: Yes")
print("Missing target values: 0")
print("Negative target values: 0")


# ============================================================
# 4. CREATE CALENDAR FEATURES
# ============================================================

# Create a new dataframe so the original processed dataset remains
# unchanged on disk.

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

# December 25 indicator.
df["is_christmas"] = (
    (df["date"].dt.month == 12)
    & (df["date"].dt.day == 25)
).astype(int)


# ============================================================
# 5. CREATE LAG FEATURES
# ============================================================

# A lag feature uses previously observed sales.
#
# lag_1: sales one day earlier
# lag_7: sales seven days earlier
# lag_14: sales fourteen days earlier
# lag_28: sales twenty-eight days earlier

df["lag_1"] = df["units_sold"].shift(1)
df["lag_7"] = df["units_sold"].shift(7)
df["lag_14"] = df["units_sold"].shift(14)
df["lag_28"] = df["units_sold"].shift(28)


# ============================================================
# 6. CREATE ROLLING-AVERAGE FEATURES
# ============================================================

# shift(1) prevents the rolling averages from including the
# current day's target value.

df["rolling_mean_7"] = (
    df["units_sold"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

df["rolling_mean_28"] = (
    df["units_sold"]
    .shift(1)
    .rolling(window=28)
    .mean()
)


# ============================================================
# 7. DEFINE FEATURES AND TARGET
# ============================================================

feature_columns = [
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "month",
    "year",
    "is_christmas",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

target_column = "units_sold"

print("\nFEATURE ENGINEERING")
print("-" * 65)
print(f"Number of features: {len(feature_columns)}")
print("Features:")
for feature in feature_columns:
    print(f"  - {feature}")


# ============================================================
# 8. REMOVE ROWS WITH MISSING ENGINEERED FEATURES
# ============================================================

# IMPORTANT:
# This must happen AFTER the lag and rolling-average features
# have been created. The first 28 rows naturally have missing
# historical features.

print("\nMISSING FEATURE DIAGNOSTICS")
print("-" * 65)

missing_features = df[feature_columns].isna().sum()

for column, count in missing_features.items():
    if count > 0:
        print(f"{column}: {count} missing values")

print(
    "\nRows before removing incomplete feature rows:",
    len(df)
)

# Keep rows where every model feature and the target are available.
df_model = df.dropna(
    subset=feature_columns + [target_column]
).copy()

df_model = df_model.reset_index(drop=True)

print("Rows after removing incomplete feature rows:", len(df_model))
print("Rows removed:", len(df) - len(df_model))

if len(df_model) < 100:
    raise ValueError(
        "Too few usable observations remain after feature engineering."
    )

if df_model[feature_columns].isna().any().any():
    raise ValueError(
        "Missing feature values remain after cleaning."
    )

print(f"First usable date: {df_model['date'].min().date()}")
print(f"Last usable date: {df_model['date'].max().date()}")


# ============================================================
# 9. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Do not randomly shuffle time-series data.
#
# The latest 56 usable observations are held out:
# - Last 28 observations: test set
# - Previous 28 observations: validation set
# - Earlier observations: training set

TEST_DAYS = 28
VALIDATION_DAYS = 28

if len(df_model) <= TEST_DAYS + VALIDATION_DAYS:
    raise ValueError(
        "The dataset is too small for the selected time-series split."
    )

train_end = len(df_model) - VALIDATION_DAYS - TEST_DAYS
validation_end = len(df_model) - TEST_DAYS

train_df = df_model.iloc[:train_end].copy()

validation_df = df_model.iloc[
    train_end:validation_end
].copy()

test_df = df_model.iloc[validation_end:].copy()

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_validation = validation_df[feature_columns]
y_validation = validation_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

print("\nCHRONOLOGICAL DATA SPLIT")
print("-" * 65)

print(
    f"Training rows: {len(train_df):,} | "
    f"{train_df['date'].min().date()} to "
    f"{train_df['date'].max().date()}"
)

print(
    f"Validation rows: {len(validation_df):,} | "
    f"{validation_df['date'].min().date()} to "
    f"{validation_df['date'].max().date()}"
)

print(
    f"Test rows: {len(test_df):,} | "
    f"{test_df['date'].min().date()} to "
    f"{test_df['date'].max().date()}"
)


# ============================================================
# 10. DEFINE EVALUATION METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    """
    Calculate MAE, RMSE, and WAPE.

    MAE:
        Mean absolute error.

    RMSE:
        Root mean squared error.

    WAPE:
        Weighted absolute percentage error.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    actual_sum = np.sum(np.abs(actual))

    if actual_sum == 0:
        wape = np.nan
    else:
        wape = (
            np.sum(np.abs(actual - predicted))
            / actual_sum
        ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE_percent": wape,
    }


def save_prediction_file(
    dates,
    actual,
    predicted,
    output_path,
    model_name
):
    """
    Save actual values and predictions to a CSV file.
    """

    prediction_df = pd.DataFrame({
        "date": pd.to_datetime(dates).to_numpy(),
        "actual_units_sold": np.asarray(actual),
        "predicted_units_sold": np.asarray(predicted),
    })

    prediction_df["prediction_error"] = (
        prediction_df["actual_units_sold"]
        - prediction_df["predicted_units_sold"]
    )

    prediction_df["absolute_error"] = (
        prediction_df["prediction_error"].abs()
    )

    prediction_df["model"] = model_name

    prediction_df.to_csv(output_path, index=False)

    return prediction_df


# ============================================================
# 11. SEASONAL NAIVE BASELINE
# ============================================================

# The baseline predicts that sales will equal sales from the
# same weekday one week earlier.
#
# This uses lag_7, which is based on previously observed sales.

print("\nSEASONAL NAIVE BASELINE")
print("-" * 65)

validation_baseline_predictions = (
    validation_df["lag_7"].to_numpy()
)

test_baseline_predictions = (
    test_df["lag_7"].to_numpy()
)

validation_baseline_metrics = calculate_metrics(
    y_validation,
    validation_baseline_predictions
)

test_baseline_metrics = calculate_metrics(
    y_test,
    test_baseline_predictions
)

print("Validation metrics:")
for metric, value in validation_baseline_metrics.items():
    print(f"  {metric}: {value:.4f}")

print("\nTest metrics:")
for metric, value in test_baseline_metrics.items():
    print(f"  {metric}: {value:.4f}")


# ============================================================
# 12. TRAIN THE RANDOM FOREST MODEL
# ============================================================

print("\nTRAINING RANDOM FOREST MODEL")
print("-" * 65)

random_forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

random_forest.fit(
    X_train,
    y_train
)

# Evaluate on validation data first.
validation_rf_predictions = random_forest.predict(
    X_validation
)

validation_rf_metrics = calculate_metrics(
    y_validation,
    validation_rf_predictions
)

print("\nValidation Random Forest metrics:")
for metric, value in validation_rf_metrics.items():
    print(f"  {metric}: {value:.4f}")


# ============================================================
# 13. RETRAIN USING TRAINING + VALIDATION DATA
# ============================================================

# The validation period is used for model assessment.
# After that assessment, refit the selected model using both
# training and validation observations. The test period remains
# untouched until final evaluation.

train_validation_df = pd.concat(
    [train_df, validation_df],
    ignore_index=True
)

X_train_validation = train_validation_df[feature_columns]
y_train_validation = train_validation_df[target_column]

final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

final_model.fit(
    X_train_validation,
    y_train_validation
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

print("\nFINAL TEST EVALUATION")
print("-" * 65)

test_rf_predictions = final_model.predict(X_test)

test_rf_metrics = calculate_metrics(
    y_test,
    test_rf_predictions
)

print("Test Random Forest metrics:")
for metric, value in test_rf_metrics.items():
    print(f"  {metric}: {value:.4f}")


# ============================================================
# 15. SAVE METRICS
# ============================================================

metrics_rows = [
    {
        "dataset": "Validation",
        "model": "Seasonal Naive",
        **validation_baseline_metrics,
    },
    {
        "dataset": "Validation",
        "model": "Random Forest",
        **validation_rf_metrics,
    },
    {
        "dataset": "Test",
        "model": "Seasonal Naive",
        **test_baseline_metrics,
    },
    {
        "dataset": "Test",
        "model": "Random Forest",
        **test_rf_metrics,
    },
]

metrics_df = pd.DataFrame(metrics_rows)

metrics_path = RESULTS_DIR / "forecasting_model_metrics.csv"

metrics_df.to_csv(
    metrics_path,
    index=False
)

print(f"\nMetrics saved to:\n{metrics_path}")


# ============================================================
# 16. SAVE VALIDATION PREDICTIONS
# ============================================================

validation_prediction_df = save_prediction_file(
    dates=validation_df["date"],
    actual=y_validation,
    predicted=validation_rf_predictions,
    output_path=RESULTS_DIR / "validation_predictions.csv",
    model_name="Random Forest",
)

baseline_validation_df = save_prediction_file(
    dates=validation_df["date"],
    actual=y_validation,
    predicted=validation_baseline_predictions,
    output_path=RESULTS_DIR / "validation_baseline_predictions.csv",
    model_name="Seasonal Naive",
)


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

test_prediction_df = save_prediction_file(
    dates=test_df["date"],
    actual=y_test,
    predicted=test_rf_predictions,
    output_path=RESULTS_DIR / "test_predictions.csv",
    model_name="Random Forest",
)

baseline_test_df = save_prediction_file(
    dates=test_df["date"],
    actual=y_test,
    predicted=test_baseline_predictions,
    output_path=RESULTS_DIR / "test_baseline_predictions.csv",
    model_name="Seasonal Naive",
)


# ============================================================
# 18. SAVE FEATURE IMPORTANCE
# ============================================================

feature_importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": final_model.feature_importances_,
})

feature_importance_df = feature_importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

importance_path = RESULTS_DIR / "feature_importance.csv"

feature_importance_df.to_csv(
    importance_path,
    index=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 65)
print(feature_importance_df.to_string(index=False))


# ============================================================
# 19. PLOT ACTUAL VS PREDICTED TEST SALES
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    test_df["date"],
    y_test.to_numpy(),
    label="Actual Sales",
    marker="o",
)

plt.plot(
    test_df["date"],
    test_baseline_predictions,
    label="Seasonal Naive",
    linestyle="--",
)

plt.plot(
    test_df["date"],
    test_rf_predictions,
    label="Random Forest",
    linestyle="-.",
)

plt.title(
    "Walmart Foods Demand Forecast: Test Period"
)

plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

forecast_figure_path = (
    FIGURES_DIR / "test_forecast_comparison.png"
)

plt.savefig(
    forecast_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nForecast comparison figure saved to:\n{forecast_figure_path}")


# ============================================================
# 20. PLOT FEATURE IMPORTANCE
# ============================================================

plot_importance_df = feature_importance_df.sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_importance_df["feature"],
    plot_importance_df["importance"],
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

importance_figure_path = (
    FIGURES_DIR / "random_forest_feature_importance.png"
)

plt.savefig(
    importance_figure_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Feature importance figure saved to:\n{importance_figure_path}")


# ============================================================
# 21. SAVE THE FINAL TRAINED MODEL
# ============================================================

model_path = MODELS_DIR / "walmart_food_random_forest.joblib"

joblib.dump(
    final_model,
    model_path
)

print(f"\nTrained model saved to:\n{model_path}")


# ============================================================
# 22. SUMMARIZE PERFORMANCE IMPROVEMENT
# ============================================================

baseline_mae = test_baseline_metrics["MAE"]
rf_mae = test_rf_metrics["MAE"]

baseline_rmse = test_baseline_metrics["RMSE"]
rf_rmse = test_rf_metrics["RMSE"]

baseline_wape = test_baseline_metrics["WAPE_percent"]
rf_wape = test_rf_metrics["WAPE_percent"]

mae_improvement = (
    (baseline_mae - rf_mae) / baseline_mae * 100
    if baseline_mae != 0
    else np.nan
)

rmse_improvement = (
    (baseline_rmse - rf_rmse) / baseline_rmse * 100
    if baseline_rmse != 0
    else np.nan
)

wape_improvement = (
    (baseline_wape - rf_wape) / baseline_wape * 100
    if baseline_wape != 0
    else np.nan
)

print("\nTEST PERFORMANCE COMPARISON")
print("-" * 65)
print(f"MAE improvement: {mae_improvement:.2f}%")
print(f"RMSE improvement: {rmse_improvement:.2f}%")
print(f"WAPE improvement: {wape_improvement:.2f}%")


# ============================================================
# 23. FINISH
# ============================================================

print("\nOUTPUT FILES")
print("-" * 65)
print(f"Metrics: {metrics_path}")
print(f"Validation predictions: {RESULTS_DIR / 'validation_predictions.csv'}")
print(f"Test predictions: {RESULTS_DIR / 'test_predictions.csv'}")
print(f"Feature importance: {importance_path}")
print(f"Forecast plot: {forecast_figure_path}")
print(f"Feature importance plot: {importance_figure_path}")
print(f"Trained model: {model_path}")

print("\nForecasting pipeline completed successfully.")