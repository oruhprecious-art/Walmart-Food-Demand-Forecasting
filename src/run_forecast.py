from pathlib import Path
import sys

import joblib
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "data" / "processed" / "foods_daily_demand.csv"
MODEL_PATH = PROJECT_DIR / "models" / "walmart_food_random_forest.joblib"
RESULTS_DIR = PROJECT_DIR / "results"

OUTPUT_PATH = RESULTS_DIR / "operational_forecast.csv"

FEATURES = [
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


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    """Load and validate the processed daily demand dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"date", "units_sold"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Keep only the columns required for forecasting.
    df = df[["date", "units_sold"]].copy()

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    )

    df["units_sold"] = pd.to_numeric(
        df["units_sold"], errors="raise"
    )

    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df["date"].isna().any():
        raise ValueError("The dataset contains missing dates.")

    if df["units_sold"].isna().any():
        raise ValueError("The dataset contains missing sales values.")

    if (df["units_sold"] < 0).any():
        raise ValueError("Negative sales values were found.")

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates were found.")

    daily_gaps = df["date"].diff().dropna()

    if not daily_gaps.eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "The dataset must contain consecutive daily observations."
        )

    return df


# ============================================================
# 3. CREATE MODEL FEATURES
# ============================================================

def create_features(df):
    """Create the same features used by the training pipeline."""

    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    df["is_christmas"] = (
        (df["date"].dt.month == 12)
        & (df["date"].dt.day == 25)
    ).astype(int)

    # Historical sales lags.
    df["lag_1"] = df["units_sold"].shift(1)
    df["lag_7"] = df["units_sold"].shift(7)
    df["lag_14"] = df["units_sold"].shift(14)
    df["lag_28"] = df["units_sold"].shift(28)

    # Shift before calculating rolling means to prevent leakage.
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

    return df


# ============================================================
# 4. GENERATE HISTORICAL ONE-DAY-AHEAD PREDICTIONS
# ============================================================

def generate_predictions(df):
    """Load the saved model and generate historical predictions."""

    model = joblib.load(MODEL_PATH)

    featured_df = create_features(df)

    # The first 28 days do not have all required lag features.
    usable_df = featured_df.dropna(
        subset=FEATURES
    ).copy()

    if usable_df.empty:
        raise ValueError(
            "Not enough historical observations to generate predictions."
        )

    predictions = model.predict(usable_df[FEATURES])

    output_df = usable_df[
        ["date", "units_sold"]
    ].copy()

    output_df["predicted_units_sold"] = predictions

    # Forecasts represent demand, so negative predictions are clipped.
    output_df["predicted_units_sold"] = (
        output_df["predicted_units_sold"].clip(lower=0)
    )

    return output_df


# ============================================================
# 5. SAVE RESULTS
# ============================================================

def save_results(output_df):
    """Save predictions to the results folder."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():
    print("=" * 60)
    print("WALMART FOODS DEMAND FORECASTING")
    print("Operational Forecasting Pipeline")
    print("=" * 60)

    print("\nStep 1: Loading and validating data...")

    df = load_data()

    print(f"Total daily observations: {len(df)}")
    print(f"First available date: {df['date'].min().date()}")
    print(f"Last available date: {df['date'].max().date()}")

    print("\nStep 2: Generating predictions...")

    output_df = generate_predictions(df)

    print("\nStep 3: Saving results...")

    save_results(output_df)

    print("\n" + "=" * 60)
    print("FORECASTING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Predictions generated: {len(output_df)}")
    print(f"First prediction date: {output_df['date'].min().date()}")
    print(f"Last prediction date: {output_df['date'].max().date()}")

    print(f"\nOutput file:\n{OUTPUT_PATH}")

    print("\nFirst five predictions:")
    print(output_df.head().to_string(index=False))

    print(
        "\nNote: These are historical one-day-ahead predictions. "
        "They are not forecasts for dates beyond the dataset."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr
        )
        sys.exit(1)