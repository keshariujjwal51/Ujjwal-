"""
================================================================================
SALES FORECASTING AND INVENTORY PLANNING USING MACHINE LEARNING
A Retail Analytics Approach
--------------------------------------------------------------------------------
Dataset : Kaggle - Store Sales: Time Series Forecasting
          https://www.kaggle.com/competitions/store-sales-time-series-forecasting

Academic project (PGDM / MCA - Business Analytics & Machine Learning).

This SINGLE Python file performs the complete pipeline:
Data -> Business Understanding -> Cleaning -> EDA -> Feature Engineering ->
Machine Learning -> Evaluation -> Forecasting -> Inventory Planning ->
Business Insights -> Recommendations -> Final Report.

It can be run in TWO modes from this same file:

  1) COMMAND-LINE / REPORT MODE (no extra dependency needed):
         python store_sales_forecasting.py
     Runs the full pipeline, prints a structured report to the console, and
     saves every CSV/PNG artifact to ./outputs/

  2) INTERACTIVE WEB DASHBOARD MODE (requires the optional `streamlit`
     package -- see requirements.txt):
         streamlit run store_sales_forecasting.py
     Opens a browser-based UI with a sidebar to configure the run and tabs
     to explore EDA charts, model comparison, forecasts, inventory
     recommendations, and business insights interactively.

All artifacts (CSVs + PNGs) are written to ./outputs/
================================================================================
"""

# ==============================================================================
# 1. BUSINESS PROBLEM
# ==============================================================================
#
# A retail grocery company operates many stores across several product
# families (categories). Management currently struggles to answer:
#   - How much of each product family will sell tomorrow / next week?
#   - How much inventory should be kept on hand to avoid stockouts, without
#     over-investing working capital in unsold stock?
#   - Which stores / product families are the most unpredictable, and
#     therefore need closer monitoring and higher safety stock?
#
# GOAL: Build a machine-learning based demand forecasting system and turn its
# output into concrete, explainable inventory recommendations that a
# non-technical store operations manager can act on.
#
# ==============================================================================

# ==============================================================================
# 2. IMPORT LIBRARIES
# ==============================================================================
import os
import sys
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# Try to import XGBoost. If unavailable, fall back gracefully (do NOT crash).
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[INFO] xgboost is not installed. Falling back to "
          "HistGradientBoostingRegressor for Model 2.")


# ==============================================================================
# 3. CONFIGURATION
# ==============================================================================
DATA_DIR = "./data"          # change this if your Kaggle CSVs live elsewhere
OUTPUT_DIR = "./outputs"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Computational-efficiency controls ---------------------------------------
# SAMPLE_MODE=True   -> fast execution on a reproducible subset (recommended
#                        for classroom / laptop use and for viva demos)
# SAMPLE_MODE=False  -> use the full Kaggle dataset (requires more RAM/time)
SAMPLE_MODE = True
MAX_STORES = 10
MAX_PRODUCT_FAMILIES = 8

# Forecast horizon for the "future" (beyond-test-set) forecast
FUTURE_HORIZON_DAYS = 14

# Inventory planning assumptions (clearly documented, not operational advice)
SERVICE_LEVEL_Z = 1.65        # ~95% service level (one-sided normal z-score)
ASSUMED_LEAD_TIME_DAYS = 3    # assumed replenishment lead time in days

REQUIRED_FILES = ["train.csv", "stores.csv", "oil.csv",
                   "holidays_events.csv", "transactions.csv"]


# ------------------------------------------------------------------------------
# Custom exceptions (used by both the CLI entry point and the Streamlit
# dashboard so each surface can present the error in its own natural way,
# instead of the pipeline calling sys.exit() directly -- which would also
# kill a running Streamlit server).
# ------------------------------------------------------------------------------
class MissingDataError(Exception):
    """Raised when one or more required Kaggle CSV files cannot be found."""
    pass


class InsufficientDataError(Exception):
    """Raised when too few rows remain after feature engineering to train on."""
    pass


# ==============================================================================
# 4. DATA LOADING
# ==============================================================================
def load_data():
    """
    Load the five core Kaggle CSV files from DATA_DIR.
    Raises MissingDataError with a clear, actionable message if a required
    file is missing, rather than crashing with an opaque traceback.
    """
    print("\n" + "=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)

    missing = [f for f in REQUIRED_FILES
               if not os.path.isfile(os.path.join(DATA_DIR, f))]
    if missing:
        msg = (f"Missing required file(s) in '{DATA_DIR}': {missing}\n"
               f"Please download the dataset from:\n"
               f"https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data\n"
               f"and place the CSV files inside '{DATA_DIR}/' before re-running.")
        print(f"[ERROR] {msg}")
        raise MissingDataError(msg)

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), parse_dates=["date"])
    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
    oil = pd.read_csv(os.path.join(DATA_DIR, "oil.csv"), parse_dates=["date"])
    holidays = pd.read_csv(os.path.join(DATA_DIR, "holidays_events.csv"), parse_dates=["date"])
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["date"])

    print(f"train.csv         : {train.shape}")
    print(f"stores.csv        : {stores.shape}")
    print(f"oil.csv           : {oil.shape}")
    print(f"holidays_events.csv: {holidays.shape}")
    print(f"transactions.csv  : {transactions.shape}")

    return train, stores, oil, holidays, transactions


def inspect_data(df, name):
    """Print a quick structural summary of a dataframe: shape, dtypes,
    missing values and duplicate rows. Purely diagnostic (no mutation)."""
    print(f"\n--- Inspecting: {name} ---")
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print("Data types   :")
    print(df.dtypes)
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    print(f"Missing vals : {'none' if miss.empty else dict(miss)}")
    print(f"Duplicates   : {df.duplicated().sum()}")


# ==============================================================================
# 5. DATA DICTIONARY
# ==============================================================================
DATA_DICTIONARY = {
    "date": "Calendar date of the observation",
    "store_nbr": "Unique identifier of the store",
    "family": "Product family / category (e.g. GROCERY I, BEVERAGES)",
    "sales": "Total units/value of sales for that store-family-date (TARGET)",
    "onpromotion": "Number of items of that family on promotion that day",
    "city": "City where the store is located",
    "state": "State/region where the store is located",
    "type": "Store type (A-E) as classified by the retailer",
    "cluster": "Store cluster grouping (stores with similar characteristics)",
    "dcoilwtico": "Daily international oil price (West Texas Intermediate, USD)",
    "transactions": "Number of transactions recorded at a store on a date",
    "holiday_type": "Type of calendar event (Holiday, Event, Transfer, etc.)",
    "is_holiday": "Engineered flag: 1 if the date is a national holiday/event",
}


def print_data_dictionary():
    print("\n" + "=" * 60)
    print("STEP 2: DATA DICTIONARY")
    print("=" * 60)
    dd = pd.DataFrame(list(DATA_DICTIONARY.items()), columns=["Field", "Description"])
    print(dd.to_string(index=False))


# ==============================================================================
# 6. DATA CLEANING
# ==============================================================================
def clean_data(train, stores, oil, holidays, transactions):
    """
    Clean each raw table. Every decision is explained via comments/prints
    rather than silently dropping data.
    """
    print("\n" + "=" * 60)
    print("STEP 3: DATA CLEANING")
    print("=" * 60)

    # --- train.csv ---
    before = len(train)
    train = train.drop_duplicates()
    # Sales cannot be negative in this dataset's business context (returns are
    # rare/aggregated); clip small negative noise to 0 instead of dropping rows,
    # to preserve the store-family-date panel structure needed for lags.
    neg_sales = (train["sales"] < 0).sum()
    if neg_sales:
        print(f"[CLEAN] Found {neg_sales} negative sales values -> clipped to 0.")
        train["sales"] = train["sales"].clip(lower=0)
    train["onpromotion"] = train["onpromotion"].fillna(0).astype(int)
    print(f"[CLEAN] train.csv: removed {before - len(train)} duplicate rows.")

    # --- stores.csv ---
    stores = stores.drop_duplicates()

    # --- oil.csv ---
    # Oil prices are missing on non-trading days (weekends/holidays). Since
    # oil price is a slow-moving macro variable, forward-fill then back-fill
    # is a defensible choice (do NOT drop dates, since that would break the
    # daily calendar needed for merging).
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    oil = oil.sort_values("date")
    missing_oil = oil["oil_price"].isnull().sum()
    oil["oil_price"] = oil["oil_price"].ffill().bfill()
    print(f"[CLEAN] oil.csv: filled {missing_oil} missing oil prices "
          f"(forward-fill then back-fill).")

    # --- holidays_events.csv ---
    # Transferred holidays (type == 'Transfer') and 'Bridge'/'Work Day' entries
    # describe calendar adjustments rather than the actual day off; we keep
    # the raw table but engineer a clean is_holiday flag during merging
    # (excluding events explicitly marked as transferred=True, since the
    # holiday is *not observed* on that original date).
    holidays = holidays.drop_duplicates()
    print(f"[CLEAN] holidays_events.csv: {holidays['transferred'].sum()} "
          f"transferred holidays identified (handled during merge).")

    # --- transactions.csv ---
    transactions = transactions.drop_duplicates()
    transactions = transactions.dropna(subset=["transactions"])

    inspect_data(train, "train (cleaned)")
    inspect_data(oil, "oil (cleaned)")

    return train, stores, oil, holidays, transactions


# ==============================================================================
# 7. DATA INTEGRATION
# ==============================================================================
def merge_data(train, stores, oil, holidays, transactions):
    """
    Build a single analytical dataset at (date, store_nbr, family) grain.
    Care is taken to avoid data leakage: we only merge information that
    would realistically be KNOWN before/at the point of prediction
    (store attributes, oil price, holiday calendar, promotion count).
    `transactions` is store-level *actual footfall* for that same day and is
    used only for EDA/business insight, NOT as a model feature, because it is
    only known after the day closes (using it as a feature would leak the
    outcome of the day we are trying to forecast).
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATA INTEGRATION")
    print("=" * 60)

    df = train.merge(stores, on="store_nbr", how="left")
    df = df.merge(oil[["date", "oil_price"]], on="date", how="left")
    df["oil_price"] = df["oil_price"].ffill().bfill()

    # Build a clean daily is_holiday flag (national scope, not transferred)
    hol = holidays[(holidays["transferred"] == False)].copy()
    hol_flag = hol.groupby("date").size().reset_index(name="holiday_count")
    hol_flag["is_holiday"] = 1
    df = df.merge(hol_flag[["date", "is_holiday"]], on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    print(f"Merged analytical dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Stores: {df['store_nbr'].nunique()} | Families: {df['family'].nunique()}")

    return df, transactions


# ==============================================================================
# SAMPLING (computational efficiency, time-series safe)
# ==============================================================================
def apply_sampling(df):
    """
    When SAMPLE_MODE=True, select a reproducible subset of COMPLETE
    store/family combinations (never random individual rows), preserving
    every date for each chosen store-family series so lag/rolling features
    and chronological splitting remain valid.
    """
    if not SAMPLE_MODE:
        print("[CONFIG] SAMPLE_MODE=False -> using the FULL dataset.")
        return df

    print(f"[CONFIG] SAMPLE_MODE=True -> sampling top {MAX_STORES} stores "
          f"x top {MAX_PRODUCT_FAMILIES} product families (by total sales), "
          f"reproducible with RANDOM_STATE={RANDOM_STATE}.")

    top_stores = (df.groupby("store_nbr")["sales"].sum()
                  .sort_values(ascending=False).head(MAX_STORES).index)
    top_families = (df.groupby("family")["sales"].sum()
                     .sort_values(ascending=False).head(MAX_PRODUCT_FAMILIES).index)

    sampled = df[df["store_nbr"].isin(top_stores) & df["family"].isin(top_families)].copy()
    print(f"[CONFIG] Sampled dataset shape: {sampled.shape}")
    return sampled


# ==============================================================================
# 8. EXPLORATORY DATA ANALYSIS
# ==============================================================================
def _savefig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    try:
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        print(f"[PLOT] saved -> {path}")
    except Exception as e:
        print(f"[WARN] could not save plot {filename}: {e}")
    finally:
        plt.close(fig)


def perform_eda(df, transactions):
    """Generate and save the required exploratory visualizations."""
    print("\n" + "=" * 60)
    print("STEP 5: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Overall sales trend
    daily = df.groupby("date")["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["date"], daily["sales"], color="steelblue", linewidth=1)
    ax.set_title("Overall Daily Sales Trend")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Sales")
    _savefig(fig, "sales_trend.png")

    # 2. Monthly sales trend
    monthly = df.copy()
    monthly["year_month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly = monthly.groupby("year_month")["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(monthly["year_month"], monthly["sales"], color="seagreen")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Year-Month"); ax.set_ylabel("Total Sales")
    ax.tick_params(axis="x", rotation=90)
    _savefig(fig, "monthly_sales.png")

    # 3. Top product families
    top_fam = df.groupby("family")["sales"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=top_fam.values, y=top_fam.index, ax=ax, palette="viridis")
    ax.set_title("Top Product Families by Total Sales")
    ax.set_xlabel("Total Sales"); ax.set_ylabel("Product Family")
    _savefig(fig, "top_products.png")

    # 4. Top stores
    top_store = df.groupby("store_nbr")["sales"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=top_store.values, y=top_store.index.astype(str), ax=ax, palette="magma")
    ax.set_title("Top Stores by Total Sales")
    ax.set_xlabel("Total Sales"); ax.set_ylabel("Store Number")
    _savefig(fig, "top_stores.png")

    # 5. Sales distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["sales"], bins=60, ax=ax, color="darkorange")
    ax.set_title("Distribution of Sales (store-family-day)")
    ax.set_xlabel("Sales"); ax.set_ylabel("Frequency")
    _savefig(fig, "sales_distribution.png")

    # 6. Promotion vs sales
    promo = df.copy()
    promo["on_promo"] = (promo["onpromotion"] > 0).map({True: "On Promotion", False: "No Promotion"})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=promo, x="on_promo", y="sales", ax=ax, estimator=np.mean,
                errorbar=None, palette="Set2")
    ax.set_title("Average Sales: Promotion vs No Promotion")
    ax.set_xlabel(""); ax.set_ylabel("Average Sales")
    _savefig(fig, "promotion_analysis.png")

    # 7. Holiday vs non-holiday sales
    fig, ax = plt.subplots(figsize=(6, 5))
    hol_map = df.copy()
    hol_map["is_holiday"] = hol_map["is_holiday"].map({1: "Holiday", 0: "Regular Day"})
    sns.barplot(data=hol_map, x="is_holiday", y="sales", ax=ax, estimator=np.mean,
                errorbar=None, palette="Set1")
    ax.set_title("Average Sales: Holiday vs Regular Day")
    ax.set_xlabel(""); ax.set_ylabel("Average Sales")
    _savefig(fig, "holiday_analysis.png")

    # 8 & 9. Store-wise / family-wise boxplots (seasonality-adjacent)
    fig, ax = plt.subplots(figsize=(10, 5))
    top10_fam = df.groupby("family")["sales"].sum().sort_values(ascending=False).head(8).index
    sns.boxplot(data=df[df["family"].isin(top10_fam)], x="family", y="sales", ax=ax, showfliers=False)
    ax.set_title("Sales Spread by Product Family (Top 8)")
    ax.set_xlabel("Product Family"); ax.set_ylabel("Sales")
    ax.tick_params(axis="x", rotation=45)
    _savefig(fig, "product_family_sales.png")

    # 10. Sales seasonality (month-of-year average)
    seas = df.copy()
    seas["month"] = seas["date"].dt.month
    seas = seas.groupby("month")["sales"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=seas, x="month", y="sales", marker="o", ax=ax, color="crimson")
    ax.set_title("Average Sales by Month (Seasonality)")
    ax.set_xlabel("Month"); ax.set_ylabel("Average Sales")
    ax.set_xticks(range(1, 13))
    _savefig(fig, "seasonality.png")

    # 11. Correlation heatmap
    num_cols = ["sales", "onpromotion", "oil_price", "is_holiday", "cluster"]
    num_cols = [c for c in num_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    _savefig(fig, "correlation_heatmap.png")

    print("EDA complete. Plots saved to", OUTPUT_DIR)


# ==============================================================================
# 9. FEATURE ENGINEERING
# ==============================================================================
def create_features(df):
    """
    Build date, lag, and rolling features at (store_nbr, family) series level.
    CRITICAL: lag/rolling windows are computed using .shift(1) BEFORE rolling,
    so that the value at time t never uses sales observed at time t itself
    (no leakage of the target into its own features).
    """
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE ENGINEERING")
    print("=" * 60)

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # --- Date features ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # --- Lag & rolling features (grouped per store-family series) ---
    grp = df.groupby(["store_nbr", "family"])["sales"]

    df["lag_1"] = grp.shift(1)
    df["lag_7"] = grp.shift(7)
    df["lag_14"] = grp.shift(14)
    df["lag_28"] = grp.shift(28)

    # Rolling stats computed on the SHIFTED series (t-1 backwards) so the
    # current day's sales never leaks into its own rolling features.
    shifted = grp.shift(1)
    df["rolling_mean_7"] = shifted.groupby([df["store_nbr"], df["family"]]).transform(
        lambda s: s.rolling(window=7, min_periods=1).mean())
    df["rolling_mean_14"] = shifted.groupby([df["store_nbr"], df["family"]]).transform(
        lambda s: s.rolling(window=14, min_periods=1).mean())
    df["rolling_mean_28"] = shifted.groupby([df["store_nbr"], df["family"]]).transform(
        lambda s: s.rolling(window=28, min_periods=1).mean())
    df["rolling_std_7"] = shifted.groupby([df["store_nbr"], df["family"]]).transform(
        lambda s: s.rolling(window=7, min_periods=1).std())
    df["rolling_std_28"] = shifted.groupby([df["store_nbr"], df["family"]]).transform(
        lambda s: s.rolling(window=28, min_periods=1).std())

    # --- Encode categoricals ---
    le_family = LabelEncoder()
    le_type = LabelEncoder()
    le_city = LabelEncoder()
    df["family_enc"] = le_family.fit_transform(df["family"])
    df["store_type_enc"] = le_type.fit_transform(df["type"])
    df["city_enc"] = le_city.fit_transform(df["city"])

    before = len(df)
    # Rows without enough history for the longest lag (28 days) cannot be
    # used for supervised training; we drop only those, not the whole
    # dataset, and we report exactly how many.
    df_model = df.dropna(subset=["lag_28", "rolling_std_28"]).reset_index(drop=True)
    print(f"[FEATURES] Dropped {before - len(df_model)} rows lacking sufficient "
          f"lag history (< 28 prior days) out of {before} total rows.")

    print(f"[FEATURES] Final feature-ready dataset shape: {df_model.shape}")

    return df, df_model, le_family, le_type, le_city


FEATURE_COLUMNS = [
    "store_nbr", "onpromotion", "oil_price", "is_holiday", "cluster",
    "year", "month", "quarter", "week", "day", "day_of_week", "day_of_year",
    "is_weekend", "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28", "family_enc", "store_type_enc", "city_enc",
]
TARGET_COLUMN = "sales"


# ==============================================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
# ==============================================================================
def prepare_model_data(df_model):
    """
    Chronological (time-based) split: 70% train / 15% validation / 15% test.
    Random splitting is NOT used because it would let the model "see" future
    dates during training and validate on the past -- an unrealistic and
    leaky evaluation for a forecasting problem. Chronological splitting
    mimics how the model will actually be used in production: trained on the
    past, evaluated on the future.
    """
    print("\n" + "=" * 60)
    print("STEP 7: TRAIN / VALIDATION / TEST SPLIT (chronological)")
    print("=" * 60)

    df_model = df_model.sort_values("date").reset_index(drop=True)
    dates = df_model["date"].sort_values().unique()
    n = len(dates)
    train_cut = dates[int(n * 0.70)]
    val_cut = dates[int(n * 0.85)]

    train_set = df_model[df_model["date"] < train_cut]
    val_set = df_model[(df_model["date"] >= train_cut) & (df_model["date"] < val_cut)]
    test_set = df_model[df_model["date"] >= val_cut]

    print(f"Train: {train_set['date'].min().date()} -> {train_set['date'].max().date()} "
          f"({len(train_set)} rows)")
    print(f"Val  : {val_set['date'].min().date()} -> {val_set['date'].max().date()} "
          f"({len(val_set)} rows)")
    print(f"Test : {test_set['date'].min().date()} -> {test_set['date'].max().date()} "
          f"({len(test_set)} rows)")

    X_train, y_train = train_set[FEATURE_COLUMNS], train_set[TARGET_COLUMN]
    X_val, y_val = val_set[FEATURE_COLUMNS], val_set[TARGET_COLUMN]
    X_test, y_test = test_set[FEATURE_COLUMNS], test_set[TARGET_COLUMN]

    return (train_set, val_set, test_set,
            X_train, y_train, X_val, y_val, X_test, y_test)


# ==============================================================================
# 11. BASELINE MODEL
# ==============================================================================
def compute_baseline(test_set):
    """
    Naive baseline: predict today's sales = 7-day rolling mean of sales
    (a simple, explainable benchmark commonly used in demand forecasting).
    Any real ML model must beat this to be considered useful.
    """
    print("\n" + "=" * 60)
    print("STEP 8: BASELINE MODEL")
    print("=" * 60)

    baseline_pred = test_set["rolling_mean_7"].fillna(test_set["lag_1"]).fillna(0)
    actual = test_set[TARGET_COLUMN]

    metrics = evaluate_predictions(actual, baseline_pred)
    print(f"Baseline (7-day rolling mean) -> MAE={metrics['MAE']:.2f}, "
          f"RMSE={metrics['RMSE']:.2f}, MAPE={metrics['MAPE']:.2f}%, R2={metrics['R2']:.3f}")
    return baseline_pred, metrics


# ==============================================================================
# 12. MACHINE LEARNING MODELS
# ==============================================================================
def train_models(X_train, y_train, X_val, y_val):
    """
    Train two tree-based ML models:
      Model 1: Random Forest Regressor
      Model 2: XGBoost Regressor (falls back to HistGradientBoostingRegressor
               if xgboost is not installed)
    Tree-based models are chosen for interpretability (feature importance),
    robustness to mixed feature scales, and strong out-of-the-box performance
    on tabular retail data.
    """
    print("\n" + "=" * 60)
    print("STEP 9: MACHINE LEARNING MODELS")
    print("=" * 60)

    models = {}

    print("[TRAIN] Random Forest Regressor ...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    if XGBOOST_AVAILABLE:
        print("[TRAIN] XGBoost Regressor ...")
        xgb = XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
            objective="reg:squarederror"
        )
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb
    else:
        print("[TRAIN] HistGradientBoostingRegressor (XGBoost fallback) ...")
        hgb = HistGradientBoostingRegressor(
            max_iter=300, max_depth=7, learning_rate=0.05,
            random_state=RANDOM_STATE
        )
        hgb.fit(X_train, y_train)
        models["HistGradientBoosting (XGB fallback)"] = hgb

    return models


# ==============================================================================
# 13. MODEL EVALUATION
# ==============================================================================
def evaluate_predictions(actual, predicted):
    """Compute MAE, RMSE, MAPE, R2. MAPE ignores rows where actual==0 to
    avoid division-by-zero distortion (common in intermittent retail demand)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    nonzero = actual != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    else:
        mape = np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def evaluate_models(models, X_test, y_test, baseline_metrics):
    """
    Evaluate all trained models on the held-out chronological test set and
    build a comparison table including the baseline.
    Model selection is NOT based on R2 alone -- MAE/RMSE are in the original
    sales units and are more directly meaningful for inventory decisions,
    while MAPE communicates relative forecast error to non-technical
    stakeholders.
    """
    print("\n" + "=" * 60)
    print("STEP 10: MODEL EVALUATION")
    print("=" * 60)

    results = {"Baseline (7-day rolling mean)": baseline_metrics}
    predictions = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None)  # sales cannot be negative
        predictions[name] = preds
        results[name] = evaluate_predictions(y_test, preds)
        print(f"{name:35s} -> MAE={results[name]['MAE']:.2f}, "
              f"RMSE={results[name]['RMSE']:.2f}, "
              f"MAPE={results[name]['MAPE']:.2f}%, R2={results[name]['R2']:.3f}")

    comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    comparison_df = comparison_df[["Model", "MAE", "RMSE", "MAPE", "R2"]]
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\n[SAVED] {OUTPUT_DIR}/model_comparison.csv")

    # Best model chosen primarily on MAE (business-relevant absolute error),
    # among the actual ML models (excluding the baseline).
    ml_only = comparison_df[comparison_df["Model"] != "Baseline (7-day rolling mean)"]
    best_model_name = ml_only.sort_values("MAE").iloc[0]["Model"]
    print(f"\n[SELECTED] Best model by MAE: {best_model_name}")

    return comparison_df, predictions, best_model_name


# ==============================================================================
# 14. FEATURE IMPORTANCE
# ==============================================================================
def analyze_feature_importance(models, best_model_name):
    """Extract and visualize feature importance for the best tree-based model."""
    print("\n" + "=" * 60)
    print("STEP 11: FEATURE IMPORTANCE")
    print("=" * 60)

    best_model = models[best_model_name]
    if not hasattr(best_model, "feature_importances_"):
        print(f"[WARN] {best_model_name} does not expose feature_importances_. Skipping.")
        return None

    fi = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    fi.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print(f"[SAVED] {OUTPUT_DIR}/feature_importance.csv")
    print(fi.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi.head(15), x="importance", y="feature", ax=ax, palette="crest")
    ax.set_title(f"Top 15 Feature Importances ({best_model_name})")
    ax.set_xlabel("Importance"); ax.set_ylabel("Feature")
    _savefig(fig, "feature_importance.png")

    return fi


# ==============================================================================
# 15. FORECASTING
# ==============================================================================
def generate_forecast(test_set, predictions, best_model_name):
    """Build and save the test-period actual-vs-predicted table and plot."""
    print("\n" + "=" * 60)
    print("STEP 12: FORECASTING (historical test-period predictions)")
    print("=" * 60)

    best_preds = predictions[best_model_name]
    result = test_set[["date", "store_nbr", "family", "sales"]].copy()
    result = result.rename(columns={"sales": "actual_sales"})
    result["predicted_sales"] = best_preds
    result.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"[SAVED] {OUTPUT_DIR}/test_predictions.csv ({len(result)} rows)")

    daily_actual = result.groupby("date")["actual_sales"].sum()
    daily_pred = result.groupby("date")["predicted_sales"].sum()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily_actual.index, daily_actual.values, label="Actual", color="black", linewidth=1.3)
    ax.plot(daily_pred.index, daily_pred.values, label="Predicted", color="tomato",
            linewidth=1.3, linestyle="--")
    ax.set_title(f"Forecast vs Actual - Test Period ({best_model_name})")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Sales")
    ax.legend()
    _savefig(fig, "forecast_vs_actual.png")

    return result


def generate_future_forecast(df, df_model, models, best_model_name,
                              le_family, le_type, le_city):
    """
    Produce a genuine FUTURE forecast (beyond the last date in the dataset)
    for FUTURE_HORIZON_DAYS days, per store-family series, using recursive
    (iterative) prediction: each day's forecast becomes the lag_1 input for
    the next day, and rolling stats are recomputed from the growing series
    of (actual history + forecasts so far).

    This is explicitly labeled as a FORECAST, not an observation -- these
    values have never been seen and are model estimates only.
    """
    print("\n" + "=" * 60)
    print("STEP 13: FUTURE FORECAST (beyond available history)")
    print("=" * 60)

    best_model = models[best_model_name]
    last_date = df["date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, FUTURE_HORIZON_DAYS + 1)]

    all_future_rows = []
    series_keys = df_model[["store_nbr", "family"]].drop_duplicates().values

    for store_nbr, family in series_keys:
        hist = df[(df["store_nbr"] == store_nbr) & (df["family"] == family)].sort_values("date")
        if len(hist) < 28:
            continue  # not enough history to build lag_28 safely

        sales_history = hist["sales"].tolist()
        last_row = hist.iloc[-1]
        oil_price = last_row["oil_price"]
        cluster = last_row["cluster"]
        store_type_enc = le_type.transform([last_row["type"]])[0]
        city_enc = le_city.transform([last_row["city"]])[0]
        family_enc = le_family.transform([family])[0]
        avg_onpromotion = hist["onpromotion"].tail(28).mean()

        for f_date in future_dates:
            lag_1 = sales_history[-1]
            lag_7 = sales_history[-7] if len(sales_history) >= 7 else np.mean(sales_history)
            lag_14 = sales_history[-14] if len(sales_history) >= 14 else np.mean(sales_history)
            lag_28 = sales_history[-28] if len(sales_history) >= 28 else np.mean(sales_history)
            roll7 = np.mean(sales_history[-7:])
            roll14 = np.mean(sales_history[-14:])
            roll28 = np.mean(sales_history[-28:])
            std7 = np.std(sales_history[-7:]) if len(sales_history) >= 2 else 0.0
            std28 = np.std(sales_history[-28:]) if len(sales_history) >= 2 else 0.0

            feat_row = {
                "store_nbr": store_nbr, "onpromotion": avg_onpromotion,
                "oil_price": oil_price, "is_holiday": 0, "cluster": cluster,
                "year": f_date.year, "month": f_date.month,
                "quarter": (f_date.month - 1) // 3 + 1, "week": f_date.isocalendar()[1],
                "day": f_date.day, "day_of_week": f_date.weekday(),
                "day_of_year": f_date.timetuple().tm_yday,
                "is_weekend": int(f_date.weekday() >= 5),
                "lag_1": lag_1, "lag_7": lag_7, "lag_14": lag_14, "lag_28": lag_28,
                "rolling_mean_7": roll7, "rolling_mean_14": roll14, "rolling_mean_28": roll28,
                "rolling_std_7": std7, "rolling_std_28": std28,
                "family_enc": family_enc, "store_type_enc": store_type_enc,
                "city_enc": city_enc,
            }
            X_future = pd.DataFrame([feat_row])[FEATURE_COLUMNS]
            pred = max(0.0, float(best_model.predict(X_future)[0]))

            all_future_rows.append({
                "date": f_date, "store_nbr": store_nbr, "family": family,
                "forecast_sales": pred, "is_future_forecast": True
            })
            sales_history.append(pred)  # feed forecast back in recursively

    future_df = pd.DataFrame(all_future_rows)
    future_df.to_csv(os.path.join(OUTPUT_DIR, "future_forecast.csv"), index=False)
    print(f"[SAVED] {OUTPUT_DIR}/future_forecast.csv ({len(future_df)} rows, "
          f"{FUTURE_HORIZON_DAYS}-day horizon)")
    print("[NOTE] future_forecast.csv contains MODEL-GENERATED FORECASTS for "
          "dates beyond the dataset's last observed date - these are NOT actual sales.")

    return future_df


# ==============================================================================
# 16. INVENTORY PLANNING
# ==============================================================================
def perform_inventory_analysis(future_df, df_model):
    """
    Convert the future demand forecast into an inventory recommendation per
    store-family combination:

        Safety Stock          = Z * demand_std * sqrt(lead_time_days)
        Recommended Inventory = Expected Demand (over lead time) + Safety Stock

    ASSUMPTIONS (explicitly stated - NOT an operational recommendation):
      - Service level Z = {z} (~95% in-stock probability, one-sided normal)
      - Assumed replenishment lead time = {lt} days
      - demand_std is estimated from each series' historical daily rolling
        volatility (rolling_std_28), a proxy for real demand variability.
    Actual inventory decisions must also account for supplier lead times,
    warehouse capacity, shelf life/perishability, minimum order quantities,
    and company inventory policy - none of which are modeled here.
    """
    print("\n" + "=" * 60)
    print("STEP 14: INVENTORY PLANNING")
    print("=" * 60)
    print(f"[ASSUMPTIONS] service_level_z={SERVICE_LEVEL_Z}, "
          f"lead_time_days={ASSUMED_LEAD_TIME_DAYS}")

    demand_std = (df_model.groupby(["store_nbr", "family"])["rolling_std_28"]
                  .mean().reset_index().rename(columns={"rolling_std_28": "demand_std"}))
    demand_std["demand_std"] = demand_std["demand_std"].fillna(0)

    horizon_demand = (future_df.groupby(["store_nbr", "family"])["forecast_sales"]
                       .sum().reset_index().rename(columns={"forecast_sales": "forecast_demand"}))

    inv = horizon_demand.merge(demand_std, on=["store_nbr", "family"], how="left")
    inv["demand_std"] = inv["demand_std"].fillna(inv["demand_std"].median())

    inv["safety_stock"] = (SERVICE_LEVEL_Z * inv["demand_std"] *
                            np.sqrt(ASSUMED_LEAD_TIME_DAYS)).round(1)
    inv["recommended_inventory"] = (inv["forecast_demand"] + inv["safety_stock"]).round(1)
    inv["forecast_demand"] = inv["forecast_demand"].round(1)

    inv = inv.sort_values("recommended_inventory", ascending=False)
    inv.to_csv(os.path.join(OUTPUT_DIR, "inventory_recommendations.csv"), index=False)
    print(f"[SAVED] {OUTPUT_DIR}/inventory_recommendations.csv ({len(inv)} rows)")
    print(inv.head(10).to_string(index=False))

    # Inventory / demand-variability visualization
    fig, ax = plt.subplots(figsize=(9, 5))
    top_var = inv.sort_values("demand_std", ascending=False).head(15)
    labels = top_var["store_nbr"].astype(str) + " | " + top_var["family"]
    sns.barplot(x=top_var["demand_std"], y=labels, ax=ax, palette="rocket")
    ax.set_title("Top 15 Store-Family Combinations by Demand Volatility")
    ax.set_xlabel("Estimated Demand Std. Dev."); ax.set_ylabel("Store | Family")
    _savefig(fig, "inventory_analysis.png")

    return inv


# ==============================================================================
# 17. BUSINESS INSIGHTS
# ==============================================================================
def generate_business_insights(df, inv_df):
    """Compute the automated business insights required for the final report."""
    print("\n" + "=" * 60)
    print("STEP 15: BUSINESS INSIGHTS")
    print("=" * 60)

    top_family = df.groupby("family")["sales"].sum().idxmax()
    top_store = df.groupby("store_nbr")["sales"].sum().idxmax()

    promo_avg = df[df["onpromotion"] > 0]["sales"].mean()
    non_promo_avg = df[df["onpromotion"] == 0]["sales"].mean()
    promo_lift_pct = ((promo_avg - non_promo_avg) / non_promo_avg) * 100 if non_promo_avg else np.nan

    holiday_avg = df[df["is_holiday"] == 1]["sales"].mean()
    regular_avg = df[df["is_holiday"] == 0]["sales"].mean()
    holiday_lift_pct = ((holiday_avg - regular_avg) / regular_avg) * 100 if regular_avg else np.nan

    monthly_avg = df.groupby(df["date"].dt.month)["sales"].mean()
    best_month, worst_month = monthly_avg.idxmax(), monthly_avg.idxmin()

    most_volatile = inv_df.sort_values("demand_std", ascending=False).iloc[0]

    insights = {
        "top_product_family": top_family,
        "top_store": int(top_store),
        "promotion_lift_pct": round(promo_lift_pct, 1),
        "holiday_lift_pct": round(holiday_lift_pct, 1),
        "best_month": int(best_month),
        "worst_month": int(worst_month),
        "most_volatile_store": int(most_volatile["store_nbr"]),
        "most_volatile_family": most_volatile["family"],
    }

    for k, v in insights.items():
        print(f"{k:24s}: {v}")

    return insights


# ==============================================================================
# 18. BUSINESS RECOMMENDATIONS
# ==============================================================================
def generate_recommendations(insights, comparison_df, best_model_name):
    """Produce 5-8 actionable recommendations derived directly from the
    computed results (not generic boilerplate)."""
    print("\n" + "=" * 60)
    print("STEP 16: BUSINESS RECOMMENDATIONS")
    print("=" * 60)

    best_mae = comparison_df[comparison_df["Model"] == best_model_name]["MAE"].values[0]

    recs = [
        f"Prioritize inventory planning around '{insights['top_product_family']}', "
        f"the highest-selling product family, to minimize stockout risk on the "
        f"biggest revenue driver.",

        f"Store #{insights['top_store']} generates the highest sales volume; consider "
        f"dedicated replenishment schedules and higher base stock levels there.",

        f"Promotions lift average sales by approximately {insights['promotion_lift_pct']}%. "
        f"Align promotional calendars with inventory build-up 3-5 days in advance "
        f"to avoid stockouts during promo periods.",

        f"Holidays change average sales by approximately {insights['holiday_lift_pct']}%. "
        f"Adjust staffing and stock ahead of national holidays/events identified "
        f"in the holiday calendar.",

        f"Month {insights['best_month']} shows the highest average demand and "
        f"month {insights['worst_month']} the lowest -- use this seasonal pattern "
        f"to plan procurement budgets across the year.",

        f"Store #{insights['most_volatile_store']} / family '{insights['most_volatile_family']}' "
        f"shows the highest demand volatility in the sample; apply a higher safety-stock "
        f"buffer and more frequent demand monitoring for this combination.",

        f"The selected forecasting model ({best_model_name}) achieves a Mean Absolute "
        f"Error of {best_mae:.1f} units on unseen future data -- use this figure to set "
        f"realistic forecast-error tolerances when automating replenishment triggers.",

        "Treat the inventory_recommendations.csv safety-stock figures as a starting "
        "point only; validate against real supplier lead times, storage capacity, and "
        "shelf-life constraints before operational rollout.",
    ]

    for i, r in enumerate(recs, 1):
        print(f"{i}. {r}")

    return recs


# ==============================================================================
# SAVE OUTPUTS (misc artifacts)
# ==============================================================================
def save_outputs(df_model):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample_cols = ["date", "store_nbr", "family", "sales", "onpromotion",
                    "oil_price", "is_holiday", "lag_1", "rolling_mean_7"]
    df_model[sample_cols].head(500).to_csv(
        os.path.join(OUTPUT_DIR, "cleaned_data_sample.csv"), index=False)
    print(f"[SAVED] {OUTPUT_DIR}/cleaned_data_sample.csv")


# ==============================================================================
# 19. FINAL SUMMARY
# ==============================================================================
def print_final_summary(df, comparison_df, best_model_name, insights, recs):
    best_row = comparison_df[comparison_df["Model"] == best_model_name].iloc[0]

    print("\n" + "=" * 60)
    print("FINAL PROJECT SUMMARY")
    print("=" * 60)
    print(f"\nDataset: Kaggle Store Sales - Time Series Forecasting"
          f"{'  (SAMPLE_MODE)' if SAMPLE_MODE else '  (FULL DATA)'}")
    print(f"Observations used: {len(df)}")
    print(f"Time Period: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"\nBest Model: {best_model_name}")
    print(f"MAE : {best_row['MAE']:.2f}")
    print(f"RMSE: {best_row['RMSE']:.2f}")
    print(f"MAPE: {best_row['MAPE']:.2f}%")
    print(f"R2  : {best_row['R2']:.3f}")
    print(f"\nTop Product Family: {insights['top_product_family']}")
    print(f"Top Store: {insights['top_store']}")
    print(f"Highest Demand Volatility: Store {insights['most_volatile_store']} "
          f"/ {insights['most_volatile_family']}")

    print("\nKey Business Findings:")
    print(f"1. Promotions lift average sales by ~{insights['promotion_lift_pct']}%.")
    print(f"2. Holidays change average sales by ~{insights['holiday_lift_pct']}%.")
    print(f"3. Month {insights['best_month']} is the strongest seasonal period; "
          f"month {insights['worst_month']} is the weakest.")

    print("\nBusiness Recommendations:")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r}")
    print("=" * 60)


# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================
def run_pipeline_core(progress_callback=None):
    """
    Execute the ENTIRE pipeline end-to-end and return every artifact needed
    by either the CLI report (main()) or the Streamlit dashboard
    (run_dashboard()), so the two front ends never duplicate pipeline logic.

    `progress_callback`, if provided, is called as progress_callback(pct, label)
    after each stage (pct in 0-100) -- used by the Streamlit dashboard to
    drive a progress bar. It is a no-op in plain CLI mode.

    Raises MissingDataError / InsufficientDataError on unrecoverable problems
    instead of calling sys.exit(), so a calling UI (e.g. Streamlit) can catch
    and display the error without killing the whole process.
    """
    def tick(pct, label):
        if progress_callback:
            progress_callback(pct, label)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_data_dictionary()
    tick(5, "Data dictionary printed")

    train, stores, oil, holidays, transactions = load_data()
    for name, d in [("train", train), ("stores", stores), ("oil", oil),
                     ("holidays_events", holidays), ("transactions", transactions)]:
        inspect_data(d, name)
    tick(15, "Data loaded & inspected")

    train, stores, oil, holidays, transactions = clean_data(
        train, stores, oil, holidays, transactions)
    tick(25, "Data cleaned")

    df, transactions = merge_data(train, stores, oil, holidays, transactions)
    df = apply_sampling(df)
    tick(35, "Data merged & sampled")

    perform_eda(df, transactions)
    tick(45, "EDA complete")

    df, df_model, le_family, le_type, le_city = create_features(df)
    tick(55, "Feature engineering complete")

    if len(df_model) < 200:
        raise InsufficientDataError(
            "Not enough rows remain after feature engineering to train a "
            "reliable model. Consider setting SAMPLE_MODE=False or "
            "increasing MAX_STORES / MAX_PRODUCT_FAMILIES.")

    save_outputs(df_model)

    (train_set, val_set, test_set,
     X_train, y_train, X_val, y_val, X_test, y_test) = prepare_model_data(df_model)
    tick(62, "Train/validation/test split done")

    baseline_pred, baseline_metrics = compute_baseline(test_set)
    tick(68, "Baseline computed")

    models = train_models(X_train, y_train, X_val, y_val)
    tick(78, "ML models trained")

    comparison_df, predictions, best_model_name = evaluate_models(
        models, X_test, y_test, baseline_metrics)
    tick(84, "Models evaluated")

    feature_importance_df = analyze_feature_importance(models, best_model_name)
    tick(88, "Feature importance computed")

    test_predictions_df = generate_forecast(test_set, predictions, best_model_name)
    tick(92, "Test-period forecast generated")

    future_df = generate_future_forecast(
        df, df_model, models, best_model_name, le_family, le_type, le_city)
    tick(96, "Future forecast generated")

    inv_df = perform_inventory_analysis(future_df, df_model)
    tick(98, "Inventory plan generated")

    insights = generate_business_insights(df, inv_df)
    recs = generate_recommendations(insights, comparison_df, best_model_name)

    print_final_summary(df, comparison_df, best_model_name, insights, recs)
    tick(100, "Done")

    return {
        "df": df,
        "df_model": df_model,
        "comparison_df": comparison_df,
        "best_model_name": best_model_name,
        "feature_importance_df": feature_importance_df,
        "test_predictions_df": test_predictions_df,
        "future_df": future_df,
        "inv_df": inv_df,
        "insights": insights,
        "recommendations": recs,
    }


def main():
    """CLI entry point: run the pipeline and exit with a clear message on failure."""
    try:
        run_pipeline_core()
    except (MissingDataError, InsufficientDataError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


# ==============================================================================
# OPTIONAL INTERACTIVE FRONTEND (Streamlit dashboard)
# ==============================================================================
# This block only runs when the file is launched with `streamlit run
# store_sales_forecasting.py`. It reuses every function above -- no pipeline
# logic is duplicated. `streamlit` is an OPTIONAL dependency: plain
# `python store_sales_forecasting.py` does not need it at all.
# ==============================================================================
def _is_streamlit_runtime():
    """Detect whether this script is being executed by `streamlit run`
    (as opposed to plain `python store_sales_forecasting.py`)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_dashboard():
    """
    Interactive web dashboard for the same pipeline, built with Streamlit.
    Launch with:  streamlit run store_sales_forecasting.py
    """
    import streamlit as st

    st.set_page_config(
        page_title="Sales Forecasting & Inventory Planning",
        page_icon="📦",
        layout="wide",
    )

    st.title("📦 Sales Forecasting and Inventory Planning")
    st.caption("A Retail Analytics Approach — Kaggle Store Sales: Time Series Forecasting")

    # ---- Sidebar: configuration & run controls ----
    global DATA_DIR, OUTPUT_DIR, SAMPLE_MODE, MAX_STORES, MAX_PRODUCT_FAMILIES, \
        FUTURE_HORIZON_DAYS, SERVICE_LEVEL_Z, ASSUMED_LEAD_TIME_DAYS

    with st.sidebar:
        st.header("⚙️ Configuration")
        DATA_DIR = st.text_input("Data directory", value=DATA_DIR,
                                  help="Folder containing train.csv, stores.csv, oil.csv, "
                                       "holidays_events.csv, transactions.csv")
        SAMPLE_MODE = st.checkbox(
            "Sample mode (fast run on a subset)", value=SAMPLE_MODE,
            help="Uncheck to use the FULL dataset (slower, needs more RAM).")
        if SAMPLE_MODE:
            MAX_STORES = st.slider("Max stores", 2, 54, MAX_STORES)
            MAX_PRODUCT_FAMILIES = st.slider("Max product families", 2, 33, MAX_PRODUCT_FAMILIES)
        FUTURE_HORIZON_DAYS = st.slider("Future forecast horizon (days)", 7, 60, FUTURE_HORIZON_DAYS)

        st.divider()
        st.subheader("Inventory Assumptions")
        SERVICE_LEVEL_Z = st.number_input(
            "Service level Z-score", value=float(SERVICE_LEVEL_Z), step=0.05,
            help="~1.65 = 95% service level, ~1.96 = ~97.5%, ~2.33 = ~99%.")
        ASSUMED_LEAD_TIME_DAYS = st.number_input(
            "Assumed lead time (days)", value=int(ASSUMED_LEAD_TIME_DAYS), step=1)

        st.divider()
        run_clicked = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

    # ---- Run pipeline on demand ----
    if run_clicked:
        progress_bar = st.progress(0, text="Starting...")
        status_box = st.empty()

        def _progress(pct, label):
            progress_bar.progress(pct / 100, text=f"{label} ({pct}%)")

        log_container = st.expander("📜 Console log", expanded=False)
        with log_container:
            log_placeholder = st.empty()

        import io
        buf = io.StringIO()
        try:
            with st.spinner("Running the full pipeline — this can take a minute..."):
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    results = run_pipeline_core(progress_callback=_progress)
                finally:
                    sys.stdout = old_stdout
                    log_placeholder.code(buf.getvalue())
            st.session_state["results"] = results
            status_box.success("✅ Pipeline completed successfully.")
        except MissingDataError as e:
            sys.stdout = old_stdout if 'old_stdout' in dir() else sys.stdout
            status_box.error(f"❌ Missing data: {e}")
            return
        except InsufficientDataError as e:
            status_box.error(f"❌ Insufficient data: {e}")
            return
        except Exception as e:
            status_box.error(f"❌ Pipeline failed: {e}")
            return

    # ---- If nothing run yet this session, try to show prior outputs ----
    results = st.session_state.get("results")
    has_saved_outputs = os.path.isdir(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0

    if results is None and not has_saved_outputs:
        st.info("👈 Configure the settings in the sidebar and click **Run Full Pipeline** "
                "to get started. Make sure the five Kaggle CSV files are in the data "
                "directory shown above.")
        return
    elif results is None and has_saved_outputs:
        st.info("Showing previously saved results from `./outputs`. "
                "Click **Run Full Pipeline** to regenerate them with current settings.")

    tabs = st.tabs(["📊 EDA", "🤖 Models", "📈 Forecast", "📦 Inventory",
                     "💡 Insights & Recommendations"])

    # --- EDA tab: show every saved chart ---
    with tabs[0]:
        st.subheader("Exploratory Data Analysis")
        eda_files = [
            ("sales_trend.png", "Overall Daily Sales Trend"),
            ("monthly_sales.png", "Monthly Sales Trend"),
            ("top_products.png", "Top Product Families"),
            ("top_stores.png", "Top Stores"),
            ("sales_distribution.png", "Sales Distribution"),
            ("promotion_analysis.png", "Promotion Impact"),
            ("holiday_analysis.png", "Holiday Impact"),
            ("product_family_sales.png", "Sales Spread by Product Family"),
            ("seasonality.png", "Monthly Seasonality"),
            ("correlation_heatmap.png", "Correlation Heatmap"),
        ]
        cols = st.columns(2)
        for i, (fname, caption) in enumerate(eda_files):
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.isfile(path):
                cols[i % 2].image(path, caption=caption, use_container_width=True)

    # --- Models tab ---
    with tabs[1]:
        st.subheader("Model Comparison")
        cmp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
        if results is not None:
            st.dataframe(results["comparison_df"], use_container_width=True)
            st.success(f"🏆 Best model selected (by MAE): **{results['best_model_name']}**")
        elif os.path.isfile(cmp_path):
            st.dataframe(pd.read_csv(cmp_path), use_container_width=True)

        st.subheader("Feature Importance")
        fi_img = os.path.join(OUTPUT_DIR, "feature_importance.png")
        fi_csv = os.path.join(OUTPUT_DIR, "feature_importance.csv")
        c1, c2 = st.columns([1, 1])
        if os.path.isfile(fi_img):
            c1.image(fi_img, use_container_width=True)
        if os.path.isfile(fi_csv):
            c2.dataframe(pd.read_csv(fi_csv), use_container_width=True, height=400)

    # --- Forecast tab ---
    with tabs[2]:
        st.subheader("Test-Period: Forecast vs Actual")
        fva_img = os.path.join(OUTPUT_DIR, "forecast_vs_actual.png")
        if os.path.isfile(fva_img):
            st.image(fva_img, use_container_width=True)

        tp_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
        if os.path.isfile(tp_path):
            st.caption("Sample of historical test-period predictions (actual, already-observed dates):")
            st.dataframe(pd.read_csv(tp_path).head(200), use_container_width=True, height=300)

        st.divider()
        st.subheader(f"⚠️ Future Forecast (next {FUTURE_HORIZON_DAYS} days — model estimate, NOT actual data)")
        fut_path = os.path.join(OUTPUT_DIR, "future_forecast.csv")
        if os.path.isfile(fut_path):
            fut_df = pd.read_csv(fut_path)
            st.dataframe(fut_df, use_container_width=True, height=300)
            daily_future = fut_df.groupby("date")["forecast_sales"].sum().reset_index()
            st.line_chart(daily_future.set_index("date"))

    # --- Inventory tab ---
    with tabs[3]:
        st.subheader("Inventory Recommendations")
        st.caption("`Recommended Inventory = Forecast Demand + Safety Stock`, where "
                   f"`Safety Stock = Z({SERVICE_LEVEL_Z}) × demand_std × √lead_time({ASSUMED_LEAD_TIME_DAYS}d)`. "
                   "Starting point only — validate against real supplier lead times, "
                   "warehouse capacity, and shelf life before operational use.")
        inv_path = os.path.join(OUTPUT_DIR, "inventory_recommendations.csv")
        if os.path.isfile(inv_path):
            st.dataframe(pd.read_csv(inv_path), use_container_width=True, height=400)
        inv_img = os.path.join(OUTPUT_DIR, "inventory_analysis.png")
        if os.path.isfile(inv_img):
            st.image(inv_img, use_container_width=True)

    # --- Insights & Recommendations tab ---
    with tabs[4]:
        if results is not None:
            insights = results["insights"]
            recs = results["recommendations"]
            st.subheader("Automated Business Insights")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top Product Family", insights["top_product_family"])
            c2.metric("Top Store", insights["top_store"])
            c3.metric("Promotion Lift", f"{insights['promotion_lift_pct']}%")
            c4.metric("Holiday Lift", f"{insights['holiday_lift_pct']}%")
            st.write(f"**Most volatile combination:** Store {insights['most_volatile_store']} "
                     f"/ {insights['most_volatile_family']}")
            st.write(f"**Strongest / weakest month:** Month {insights['best_month']} "
                     f"/ Month {insights['worst_month']}")

            st.subheader("Business Recommendations")
            for i, r in enumerate(recs, 1):
                st.markdown(f"**{i}.** {r}")
        else:
            st.info("Run the pipeline once in this session to see live insights and "
                    "recommendations here (these are computed in-memory, not saved to disk).")


if __name__ == "__main__":
    if _is_streamlit_runtime():
        run_dashboard()
    else:
        main()
