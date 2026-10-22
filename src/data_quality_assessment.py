from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sales_file = DATA_DIR / "sales_train_validation.csv"
calendar_file = DATA_DIR / "calendar.csv"

print("=" * 65)
print("WALMART DEMAND FORECASTING")
print("DETAILED DATA QUALITY ASSESSMENT")
print("=" * 65)

# ==================================================
# 2. LOAD DATA
# ==================================================

sales = pd.read_csv(sales_file)
calendar = pd.read_csv(calendar_file)

# Identify daily sales columns.
sales_columns = [
    col for col in sales.columns
    if col.startswith("d_")
]

print("\n1. DATASET DIMENSIONS")
print("-" * 45)

print("Sales rows:", sales.shape[0])
print("Sales columns:", sales.shape[1])
print("Daily sales observations:", len(sales_columns))
print("Calendar rows:", calendar.shape[0])

# ==================================================
# 3. CHECK PRODUCT METADATA
# ==================================================

metadata_columns = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

metadata_missing = sales[metadata_columns].isnull().sum()

print("\n2. MISSING PRODUCT METADATA")
print("-" * 45)
print(metadata_missing)

print(
    "\nDuplicate product-store combinations:",
    sales.duplicated(["item_id", "store_id"]).sum()
)

# ==================================================
# 4. CHECK DAILY SALES VALUES
# ==================================================

daily_sales = sales[sales_columns]

print("\n3. DAILY SALES QUALITY")
print("-" * 45)

print("Total sales cells:", daily_sales.size)

print(
    "Missing sales cells:",
    int(daily_sales.isnull().sum().sum())
)

print(
    "Negative sales cells:",
    int((daily_sales < 0).sum().sum())
)

print(
    "Zero-sales cells:",
    int((daily_sales == 0).sum().sum())
)

print(
    "Total units sold across recorded observations:",
    int(daily_sales.sum().sum())
)

print("\nDaily sales summary:")
print(daily_sales.describe().round(3))

# ==================================================
# 5. CHECK CALENDAR QUALITY
# ==================================================

calendar["date"] = pd.to_datetime(
    calendar["date"],
    errors="coerce"
)

print("\n4. CALENDAR QUALITY")
print("-" * 45)

print(
    "Invalid dates:",
    int(calendar["date"].isnull().sum())
)

print(
    "Duplicate calendar dates:",
    int(calendar["date"].duplicated().sum())
)

print(
    "Duplicate day identifiers:",
    int(calendar["d"].duplicated().sum())
)

print(
    "Missing calendar day identifiers:",
    int(calendar["d"].isnull().sum())
)

print(
    "Missing event_name_1:",
    int(calendar["event_name_1"].isnull().sum())
)

print(
    "Dates covered:",
    calendar["date"].min().date(),
    "to",
    calendar["date"].max().date()
)

# ==================================================
# 6. CATEGORY-LEVEL SALES
# ==================================================

print("\n5. CATEGORY-LEVEL SALES")
print("-" * 45)

category_sales = daily_sales.sum(axis=1).groupby(
    sales["cat_id"]
).sum()

category_summary = pd.DataFrame({
    "total_units_sold": category_sales,
    "product_count": sales.groupby("cat_id")[
        "item_id"
    ].nunique()
})

print(category_summary)

category_summary.to_csv(
    RESULTS_DIR / "category_sales_summary.csv"
)

# ==================================================
# 7. CHECK DATE ALIGNMENT
# ==================================================

calendar_days = set(calendar["d"].dropna())
sales_days = set(sales_columns)

print("\n6. DATE ALIGNMENT")
print("-" * 45)

print(
    "Sales days missing from calendar:",
    len(sales_days - calendar_days)
)

print(
    "Calendar days without sales columns:",
    len(calendar_days - sales_days)
)

# ==================================================
# 8. SAVE QUALITY SUMMARY
# ==================================================

quality_summary = pd.DataFrame({
    "measure": [
        "sales_rows",
        "sales_columns",
        "daily_sales_observations",
        "total_sales_cells",
        "missing_sales_cells",
        "negative_sales_cells",
        "zero_sales_cells",
        "duplicate_product_store_combinations",
        "invalid_calendar_dates",
        "duplicate_calendar_dates",
        "duplicate_calendar_day_identifiers"
    ],
    "value": [
        sales.shape[0],
        sales.shape[1],
        len(sales_columns),
        daily_sales.size,
        int(daily_sales.isnull().sum().sum()),
        int((daily_sales < 0).sum().sum()),
        int((daily_sales == 0).sum().sum()),
        int(sales.duplicated(["item_id", "store_id"]).sum()),
        int(calendar["date"].isnull().sum()),
        int(calendar["date"].duplicated().sum()),
        int(calendar["d"].duplicated().sum())
    ]
})

quality_summary.to_csv(
    RESULTS_DIR / "data_quality_summary.csv",
    index=False
)

print("\n7. OUTPUT FILES")
print("-" * 45)

print("Saved:", RESULTS_DIR / "data_quality_summary.csv")
print("Saved:", RESULTS_DIR / "category_sales_summary.csv")

print("\nData quality assessment completed.")