from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE = DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"


# ==================================================
# 2. LOAD DATA
# ==================================================

print("=" * 60)
print("WALMART DEMAND FORECASTING")
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

sales = pd.read_csv(SALES_FILE)
calendar = pd.read_csv(CALENDAR_FILE)

calendar["date"] = pd.to_datetime(calendar["date"])

sales_columns = [
    column for column in sales.columns
    if column.startswith("d_")
]

# Match calendar dates to the daily sales columns.
calendar_sales = calendar[
    calendar["d"].isin(sales_columns)
].copy()

calendar_sales = (
    calendar_sales
    .sort_values("date")
    .reset_index(drop=True)
)

# ==================================================
# 3. AGGREGATE DAILY SALES
# ==================================================

daily_totals = sales[sales_columns].sum(axis=0)

daily_sales = pd.DataFrame({
    "d": daily_totals.index,
    "total_sales": daily_totals.values
})

daily_sales = daily_sales.merge(
    calendar_sales[["d", "date", "weekday", "month", "year"]],
    on="d",
    how="left",
    validate="one_to_one"
)

daily_sales = daily_sales.sort_values("date")

daily_sales.to_csv(
    RESULTS_DIR / "daily_total_sales.csv",
    index=False
)

print("\n1. DAILY SALES SUMMARY")
print("-" * 40)

print("Number of dates:", len(daily_sales))
print("First date:", daily_sales["date"].min().date())
print("Last date:", daily_sales["date"].max().date())

print("\nDaily sales statistics:")
print(daily_sales["total_sales"].describe().round(2))


# ==================================================
# 4. AGGREGATE SALES BY CATEGORY
# ==================================================

category_daily = sales[
    ["cat_id"] + sales_columns
].groupby("cat_id")[sales_columns].sum()

category_daily = category_daily.T

category_daily.index.name = "d"

category_daily = category_daily.reset_index()

category_daily = category_daily.merge(
    calendar_sales[["d", "date", "weekday", "month", "year"]],
    on="d",
    how="left",
    validate="one_to_one"
)

category_daily = category_daily.sort_values("date")

category_daily.to_csv(
    RESULTS_DIR / "daily_category_sales.csv",
    index=False
)

print("\n2. CATEGORY SALES SUMMARY")
print("-" * 40)

category_names = ["FOODS", "HOUSEHOLD", "HOBBIES"]

for category in category_names:
    print(f"\n{category}")
    print("Total units:", int(category_daily[category].sum()))
    print("Average daily sales:", round(category_daily[category].mean(), 2))
    print("Daily standard deviation:", round(category_daily[category].std(), 2))


# ==================================================
# 5. PLOT TOTAL DAILY SALES
# ==================================================

plt.figure(figsize=(14, 6))

plt.plot(
    daily_sales["date"],
    daily_sales["total_sales"]
)

plt.title("Walmart Total Daily Sales")
plt.xlabel("Date")
plt.ylabel("Units Sold")

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "total_daily_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 6. PLOT CATEGORY SALES
# ==================================================

plt.figure(figsize=(14, 6))

for category in category_names:
    plt.plot(
        category_daily["date"],
        category_daily[category],
        label=category
    )

plt.title("Daily Sales by Product Category")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "category_daily_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 7. WEEKDAY ANALYSIS
# ==================================================

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_summary = (
    daily_sales
    .groupby("weekday")["total_sales"]
    .mean()
    .reindex(weekday_order)
)

weekday_summary.to_csv(
    RESULTS_DIR / "weekday_sales_summary.csv",
    header=["average_daily_sales"]
)

print("\n3. AVERAGE SALES BY WEEKDAY")
print("-" * 40)

print(weekday_summary.round(2))


# ==================================================
# 8. MONTHLY ANALYSIS
# ==================================================

daily_sales["year_month"] = (
    daily_sales["date"].dt.to_period("M").astype(str)
)

monthly_sales = (
    daily_sales
    .groupby("year_month")["total_sales"]
    .sum()
    .reset_index()
)

monthly_sales.to_csv(
    RESULTS_DIR / "monthly_sales_summary.csv",
    index=False
)

plt.figure(figsize=(14, 6))

plt.plot(
    monthly_sales["year_month"],
    monthly_sales["total_sales"]
)

plt.title("Walmart Monthly Total Sales")
plt.xlabel("Month")
plt.ylabel("Units Sold")

# Display selected month labels to improve readability.
tick_positions = list(
    range(0, len(monthly_sales), 6)
)

plt.xticks(
    tick_positions,
    monthly_sales["year_month"].iloc[tick_positions],
    rotation=45
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "monthly_total_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 9. SAVE MONTHLY CATEGORY SALES
# ==================================================

category_daily["year_month"] = (
    category_daily["date"].dt.to_period("M").astype(str)
)

monthly_category_sales = (
    category_daily
    .groupby("year_month")[category_names]
    .sum()
    .reset_index()
)

monthly_category_sales.to_csv(
    RESULTS_DIR / "monthly_category_sales.csv",
    index=False
)


# ==================================================
# 10. COMPLETION
# ==================================================

print("\n4. OUTPUT FILES")
print("-" * 40)

print("Daily total sales:")
print(RESULTS_DIR / "daily_total_sales.csv")

print("\nDaily category sales:")
print(RESULTS_DIR / "daily_category_sales.csv")

print("\nMonthly sales:")
print(RESULTS_DIR / "monthly_sales_summary.csv")

print("\nFigures saved to:")
print(FIGURES_DIR)

print("\nExploratory analysis completed successfully.")