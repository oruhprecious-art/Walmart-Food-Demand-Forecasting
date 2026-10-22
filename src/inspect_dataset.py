from pathlib import Path
import pandas as pd

# ==========================================
# 1. DEFINE PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw"

sales_file = DATA_DIR / "sales_train_validation.csv"
calendar_file = DATA_DIR / "calendar.csv"

print("=" * 60)
print("WALMART DEMAND FORECASTING")
print("INITIAL DATA INSPECTION")
print("=" * 60)

# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================

for file_path in [sales_file, calendar_file]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            "Please check the file location."
        )

print("\nRequired files found successfully.")

# ==========================================
# 3. LOAD SALES DATA
# ==========================================

sales = pd.read_csv(sales_file)

print("\n--- SALES DATA ---")
print("Rows and columns:", sales.shape)

print("\nColumn names:")
print(sales.columns[:10].tolist())

print("\nFirst five records:")
print(sales.iloc[:5, :10].to_string(index=False))

# ==========================================
# 4. LOAD CALENDAR DATA
# ==========================================

calendar = pd.read_csv(calendar_file)

calendar["date"] = pd.to_datetime(calendar["date"])

print("\n--- CALENDAR DATA ---")
print("Rows and columns:", calendar.shape)

print("\nCalendar columns:")
print(calendar.columns.tolist())

print("\nFirst five records:")
print(calendar.head().to_string(index=False))

print("\nCalendar date range:")
print("Start date:", calendar["date"].min().date())
print("End date:", calendar["date"].max().date())

# ==========================================
# 5. INSPECT PRODUCT INFORMATION
# ==========================================

print("\n--- PRODUCT INFORMATION ---")

print("Product-store series:", len(sales))
print("Unique products:", sales["item_id"].nunique())
print("Unique stores:", sales["store_id"].nunique())

print("\nProducts per category:")
print(sales.groupby("cat_id")["item_id"].nunique())

print("\nProducts per department:")
print(sales.groupby("dept_id")["item_id"].nunique())

# ==========================================
# 6. CHECK DATA QUALITY
# ==========================================

print("\n--- DATA QUALITY ---")

print("\nMissing values in product metadata:")
print(sales.iloc[:, :6].isnull().sum())

print("\nDuplicate product-store combinations:")
print(sales.duplicated(["item_id", "store_id"]).sum())

print("\nMissing values in calendar:")
print(calendar.isnull().sum())

print("\nInspection completed successfully.")