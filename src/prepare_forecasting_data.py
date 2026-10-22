"""
Walmart Food Demand Forecasting
Data Preparation Pipeline

Purpose:
    1. Load raw sales and calendar data.
    2. Select the Foods category.
    3. Aggregate daily sales across available stores and products.
    4. Join calendar information.
    5. Validate the processed time series.
    6. Save the processed dataset.

This script prepares data only.
Model training belongs in train_forecasting_models.py.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SALES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sales_train_validation.csv"
)

CALENDAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "calendar.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "foods_daily_demand.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("=" * 65)
print("WALMART FOOD DEMAND DATA PREPARATION")
print("=" * 65)

print("\nSTEP 1: Loading raw datasets...")

if not SALES_PATH.exists():
    raise FileNotFoundError(
        f"Sales dataset not found:\n{SALES_PATH}"
    )

if not CALENDAR_PATH.exists():
    raise FileNotFoundError(
        f"Calendar dataset not found:\n{CALENDAR_PATH}"
    )

sales = pd.read_csv(SALES_PATH)
calendar = pd.read_csv(CALENDAR_PATH)

print("Sales dataset shape:", sales.shape)
print("Calendar dataset shape:", calendar.shape)


# ============================================================
# 3. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\nSTEP 2: Validating required columns...")

sales_required = ["id", "item_id", "cat_id", "store_id"]
calendar_required = ["d", "date"]

missing_sales_columns = [
    col for col in sales_required if col not in sales.columns
]

missing_calendar_columns = [
    col for col in calendar_required if col not in calendar.columns
]

if missing_sales_columns:
    raise ValueError(
        f"Missing sales columns: {missing_sales_columns}"
    )

if missing_calendar_columns:
    raise ValueError(
        f"Missing calendar columns: {missing_calendar_columns}"
    )

# Daily sales columns are identifiers such as d_1, d_2, ..., d_1913.
sales_columns = [
    col for col in sales.columns
    if col.startswith("d_") and col[2:].isdigit()
]

if not sales_columns:
    raise ValueError(
        "No daily sales columns such as d_1 or d_1913 were found."
    )

# Sort sales columns numerically, not alphabetically.
sales_columns = sorted(
    sales_columns,
    key=lambda col: int(col.split("_")[1])
)

print("Number of daily sales columns:", len(sales_columns))


# ============================================================
# 4. SELECT FOODS CATEGORY
# ============================================================

print("\nSTEP 3: Selecting Foods-category products...")

foods = sales.loc[
    sales["cat_id"].eq("FOODS")
].copy()

if foods.empty:
    raise ValueError(
        "No records were found for cat_id == 'FOODS'."
    )

print("Foods product/store records:", len(foods))
print("Unique Foods products:", foods["item_id"].nunique())
print("Unique stores:", foods["store_id"].nunique())


# ============================================================
# 5. AGGREGATE DAILY SALES
# ============================================================

print("\nSTEP 4: Aggregating daily Foods sales...")

# The raw dataset is in wide format, with one column per day.
# Sum all selected Foods product/store sales for each day.

daily_totals = (
    foods[sales_columns]
    .apply(pd.to_numeric, errors="coerce")
    .sum(axis=0)
)

daily = daily_totals.rename("units_sold").reset_index()

daily = daily.rename(columns={"index": "d"})

daily["units_sold"] = pd.to_numeric(
    daily["units_sold"],
    errors="raise"
)

print("Daily aggregate observations:", len(daily))
print("Total Foods units sold:", int(daily["units_sold"].sum()))


# ============================================================
# 6. PREPARE CALENDAR DATA
# ============================================================

print("\nSTEP 5: Preparing calendar information...")

calendar["date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)

if calendar["date"].isna().any():
    raise ValueError(
        "Invalid dates were found in the calendar dataset."
    )

if calendar["d"].duplicated().any():
    raise ValueError(
        "Duplicate day identifiers were found in the calendar."
    )

calendar_columns = [
    col for col in [
        "d",
        "date",
        "weekday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
    ]
    if col in calendar.columns
]

calendar_subset = calendar[calendar_columns].copy()


# ============================================================
# 7. JOIN SALES TO CALENDAR
# ============================================================

print("\nSTEP 6: Joining sales to calendar...")

df = daily.merge(
    calendar_subset,
    on="d",
    how="left",
    validate="one_to_one",
    indicator=True
)

unmatched = int((df["_merge"] != "both").sum())

if unmatched > 0:
    raise ValueError(
        f"{unmatched} sales days did not match calendar records."
    )

df = df.drop(columns="_merge")

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 8. CREATE CALENDAR FEATURES
# ============================================================

print("\nSTEP 7: Creating calendar features...")

df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_month"] = df["date"].dt.day
df["month_number"] = df["date"].dt.month
df["year_number"] = df["date"].dt.year


# ============================================================
# 9. DATA QUALITY VALIDATION
# ============================================================

print("\nSTEP 8: Validating the processed dataset...")

if df["date"].duplicated().any():
    raise ValueError(
        "Duplicate dates were found after aggregation."
    )

if df["units_sold"].isna().any():
    raise ValueError(
        "Missing daily sales values were found."
    )

if (df["units_sold"] < 0).any():
    raise ValueError(
        "Negative daily sales values were found."
    )

date_gaps = (
    df["date"].diff().dropna().dt.days
)

gap_count = int((date_gaps != 1).sum())

if gap_count > 0:
    raise ValueError(
        f"Found {gap_count} gaps in the daily date sequence."
    )

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique dates:", df["date"].nunique())
print("Date gaps:", gap_count)
print("Missing sales values:", df["units_sold"].isna().sum())
print("Negative sales values:", (df["units_sold"] < 0).sum())
print("First date:", df["date"].min().date())
print("Last date:", df["date"].max().date())


# ============================================================
# 10. SAVE PROCESSED DATASET
# ============================================================

print("\nSTEP 9: Saving the processed dataset...")

df.to_csv(OUTPUT_PATH, index=False)

print("Processed dataset saved to:")
print(OUTPUT_PATH)

print("\nPreview:")
print(df.head().to_string(index=False))

print("\n" + "=" * 65)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 65)