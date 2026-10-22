"""
Walmart Food Demand Forecasting
Daily Demand Outlier Analysis

Purpose:
    Identify unusually low and high daily food sales using the
    Interquartile Range (IQR) method.

Important:
    Flagged observations are investigated, not automatically deleted.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "foods_daily_demand.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

print("=" * 65)
print("WALMART FOOD DEMAND OUTLIER ANALYSIS")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = ["date", "units_sold"]

if not all(column in df.columns for column in required_columns):
    raise ValueError(
        "Dataset must contain 'date' and 'units_sold' columns."
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["units_sold"] = pd.to_numeric(
    df["units_sold"],
    errors="coerce"
)

df = df.dropna(
    subset=["date", "units_sold"]
).sort_values("date").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid observations remain.")

if (df["units_sold"] < 0).any():
    raise ValueError(
        "Negative daily sales detected. Investigate data quality."
    )

print("\nDataset summary")
print("-" * 45)
print("Observations:", len(df))
print("First date:", df["date"].min().date())
print("Last date:", df["date"].max().date())
print("Minimum daily sales:", df["units_sold"].min())
print("Maximum daily sales:", df["units_sold"].max())
print("Mean daily sales:", round(df["units_sold"].mean(), 2))
print("Median daily sales:", round(df["units_sold"].median(), 2))


# ============================================================
# 3. CALCULATE IQR OUTLIER BOUNDARIES
# ============================================================

print("\nCalculating IQR boundaries...")

q1 = df["units_sold"].quantile(0.25)
q3 = df["units_sold"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Daily sales cannot be negative, so use zero as the practical
# lower boundary when identifying unusually low non-negative sales.
practical_lower_bound = max(0, lower_bound)

df["outlier_type"] = "Within IQR bounds"

df.loc[
    df["units_sold"] < practical_lower_bound,
    "outlier_type"
] = "Unusually low"

df.loc[
    df["units_sold"] > upper_bound,
    "outlier_type"
] = "Unusually high"

outliers = df[
    df["outlier_type"] != "Within IQR bounds"
].copy()

print("\nIQR results")
print("-" * 45)
print(f"First quartile (Q1): {q1:.2f}")
print(f"Third quartile (Q3): {q3:.2f}")
print(f"Interquartile range: {iqr:.2f}")
print(f"Lower IQR boundary: {lower_bound:.2f}")
print(f"Upper IQR boundary: {upper_bound:.2f}")
print(f"Practical lower boundary: {practical_lower_bound:.2f}")

print("\nOutlier counts")
print("-" * 45)
print("Total observations:", len(df))
print("Flagged observations:", len(outliers))
print(
    "Unusually high:",
    int((df["outlier_type"] == "Unusually high").sum())
)
print(
    "Unusually low:",
    int((df["outlier_type"] == "Unusually low").sum())
)


# ============================================================
# 4. SAVE FLAGGED OBSERVATIONS
# ============================================================

outlier_path = RESULTS_DIR / "daily_sales_outlier_analysis.csv"

outliers.to_csv(outlier_path, index=False)

print("\nFlagged observations saved to:")
print(outlier_path)

print("\nFirst 20 flagged observations")

if outliers.empty:
    print("No observations were flagged by the IQR method.")
else:
    print(
        outliers[
            ["date", "units_sold", "outlier_type"]
        ].head(20).to_string(index=False)
    )


# ============================================================
# 5. CREATE DAILY SALES DISTRIBUTION FIGURE
# ============================================================

plt.figure(figsize=(10, 6))

plt.boxplot(
    df["units_sold"],
    vert=True
)

plt.title("Distribution of Daily Walmart Foods Sales")
plt.ylabel("Daily Units Sold")
plt.tight_layout()

boxplot_path = FIGURES_DIR / "daily_sales_outlier_boxplot.png"

plt.savefig(
    boxplot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nBoxplot saved to:")
print(boxplot_path)


# ============================================================
# 6. CREATE DAILY SALES TREND WITH FLAGGED OBSERVATIONS
# ============================================================

plt.figure(figsize=(13, 6))

plt.plot(
    df["date"],
    df["units_sold"],
    label="Daily Sales"
)

if not outliers.empty:
    plt.scatter(
        outliers["date"],
        outliers["units_sold"],
        label="IQR-Flagged Observations"
    )

plt.title("Daily Walmart Foods Sales and IQR-Flagged Observations")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

trend_path = FIGURES_DIR / "daily_sales_outlier_trend.png"

plt.savefig(
    trend_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDaily sales trend saved to:")
print(trend_path)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("OUTLIER ANALYSIS COMPLETED")
print("=" * 65)

print("Flagged observations:", len(outliers))
print("Original observations retained:", len(df))
print("No observations were deleted by this analysis.")

print("\nGenerated artifacts:")
print("1.", outlier_path)
print("2.", boxplot_path)
print("3.", trend_path)

print("\nReview flagged observations before deciding how to handle them.")