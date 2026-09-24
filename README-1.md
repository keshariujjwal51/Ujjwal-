# Sales Forecasting and Inventory Planning Using Machine Learning
### A Retail Analytics Approach

An end-to-end, single-file academic Data Science / ML project (PGDM / MCA – Business Analytics) built on the **Kaggle Store Sales – Time Series Forecasting** dataset.

The pipeline covers: **Data → Business Understanding → Cleaning → EDA → Feature Engineering → ML → Evaluation → Forecasting → Inventory Planning → Business Recommendations.**

---

## 1. Business Problem

A retail grocery company needs to forecast future product demand accurately so it can improve inventory planning, reduce stockouts, minimize overstocking, and understand the factors that drive sales.

The project answers questions such as:
- What is the overall / seasonal sales trend?
- Which product families and stores drive the most sales?
- How do promotions, holidays, and oil prices affect sales?
- How accurately can future demand be predicted?
- What inventory level should be held per store/product family?

---

## 2. Project Structure

```
project/
├── store_sales_forecasting.py   # the entire project (single file)
├── requirements.txt
├── README.md
├── data/                        # place Kaggle CSVs here
│   ├── train.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   └── transactions.csv
└── outputs/                     # auto-created when the script runs
    ├── cleaned_data_sample.csv
    ├── model_comparison.csv
    ├── test_predictions.csv
    ├── future_forecast.csv
    ├── feature_importance.csv
    ├── inventory_recommendations.csv
    └── *.png                    # all EDA / model / inventory plots
```

---

## 3. Setup

### 3.1 Install dependencies
```bash
pip install -r requirements.txt
```
> If `xgboost` fails to install on your system, the script will still run — it automatically falls back to `HistGradientBoostingRegressor`.

### 3.2 Download the dataset
1. Go to the Kaggle competition page:
   https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
2. Download `train.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`.
3. Place all five files inside a `data/` folder next to `store_sales_forecasting.py`.

(You can change the input location by editing `DATA_DIR` near the top of the script.)

---

## 4. Running the Project

The same file supports **two ways to run it**:

### 4.1 Command-line / report mode (no extra dependency)
```bash
python store_sales_forecasting.py
```
Prints a full, structured console log (data inspection → cleaning → EDA → feature engineering → model training → evaluation → forecasting → inventory planning → business insights → final summary) and saves all artifacts to `outputs/`.

### 4.2 Interactive web dashboard mode (requires `streamlit`)
```bash
streamlit run store_sales_forecasting.py
```
Opens a browser-based UI (default: http://localhost:8501) with:
- A **sidebar** to set the data directory, toggle `SAMPLE_MODE`, adjust the sample size, the future-forecast horizon, and the inventory service-level/lead-time assumptions
- A **"Run Full Pipeline"** button that executes the exact same functions as the CLI mode, with a live progress bar and console log
- Tabs to explore results interactively:
  - **📊 EDA** — all exploratory charts
  - **🤖 Models** — model comparison table and feature importance
  - **📈 Forecast** — test-period actual-vs-predicted chart/table, plus the labeled future forecast
  - **📦 Inventory** — inventory recommendations table and demand-volatility chart
  - **💡 Insights & Recommendations** — automated business insights and the 8 recommendations

If you reopen the dashboard later without re-running the pipeline, it will automatically show the most recent results saved in `./outputs/` from a prior CLI or dashboard run.

> The dashboard is purely an optional front end — it calls the same `run_pipeline_core()` function used by the CLI, so there is no duplicated logic and no risk of the two modes producing different results.

### Configuration (top of the script)
| Variable | Purpose |
|---|---|
| `DATA_DIR` | Path to the folder containing the Kaggle CSVs |
| `OUTPUT_DIR` | Where results are saved (default `./outputs`) |
| `SAMPLE_MODE` | `True` = fast run on a reproducible subset; `False` = full dataset |
| `MAX_STORES`, `MAX_PRODUCT_FAMILIES` | Size of the subset used when `SAMPLE_MODE=True` |
| `FUTURE_HORIZON_DAYS` | How many days ahead to forecast beyond the dataset (default 14) |
| `SERVICE_LEVEL_Z`, `ASSUMED_LEAD_TIME_DAYS` | Inventory safety-stock assumptions |
| `RANDOM_STATE` | Seed for reproducibility |

`SAMPLE_MODE=True` selects the top stores and product families by total sales (never random individual rows), preserving the full chronological history of each series so lag/rolling features remain valid.

---

## 5. Methodology Summary

- **Modeling grain**: (date, store, product family) — the native grain of `train.csv`.
- **No data leakage**: lag and rolling features are built on a `.shift(1)` series, so a day's own sales never feed into its own features. `transactions.csv` is used only for exploratory analysis, never as a model input, since it is only known *after* the day ends.
- **Chronological train/validation/test split** (70% / 15% / 15%) — never random — since this is a forecasting problem and the model must be evaluated the way it will actually be used: trained on the past, tested on the future.
- **Baseline**: 7-day rolling mean naive forecast.
- **ML models**: Random Forest Regressor and XGBoost Regressor (falls back to `HistGradientBoostingRegressor` if XGBoost isn't installed).
- **Evaluation**: MAE, RMSE, MAPE, R² — with the best model chosen by MAE (business-relevant absolute error), not R² alone.
- **Future forecast**: generated recursively day-by-day beyond the dataset's last date, clearly labeled as a forecast, never presented as an actual observation.
- **Inventory planning**: `Recommended Inventory = Forecast Demand + Safety Stock`, where `Safety Stock = Z × demand_std × √lead_time`. All assumptions (service level, lead time) are explicitly stated — this is a starting point, not an operational decision.

---

## 6. Key Outputs

| File | Description |
|---|---|
| `model_comparison.csv` | Baseline vs Random Forest vs XGBoost on MAE/RMSE/MAPE/R² |
| `feature_importance.csv` / `.png` | Which variables drive sales the most |
| `test_predictions.csv` | Actual vs predicted sales on the historical held-out test period |
| `forecast_vs_actual.png` | Visual comparison of predicted vs actual test-period sales |
| `future_forecast.csv` | Genuine forward-looking forecast (beyond the dataset), explicitly flagged |
| `inventory_recommendations.csv` | Forecast demand, demand volatility, safety stock, recommended inventory per store/family |
| `inventory_analysis.png` | Store/family combinations with the highest demand volatility |
| Various EDA `.png` files | Sales trend, seasonality, promotion/holiday impact, correlations, etc. |

---

## 7. Limitations & Assumptions

- Inventory recommendations use a simplified safety-stock formula and do **not** account for supplier lead-time variability, warehouse capacity, minimum order quantities, perishability, or company-specific inventory policy.
- `SAMPLE_MODE=True` trains on a subset of stores/families for speed; set it to `False` for full-dataset results (requires more time/RAM).
- The future forecast is a model estimate, not a guarantee — accuracy degrades further into the horizon since each day's forecast is recursively fed into the next.

---

## 8. Possible Viva Questions

1. Why is a chronological split used instead of a random `train_test_split`?
2. How exactly is data leakage prevented in the lag/rolling features?
3. Why is `transactions.csv` excluded from the model's feature set?
4. Why is MAE preferred over R² when selecting the best model for demand forecasting?
5. How is the future forecast generated without any ground-truth future data?
6. How is safety stock calculated, and what does the service-level Z-score represent?
7. Why does `SAMPLE_MODE` select whole store/family series rather than random rows?
8. What would change if the model forecasted at store-level instead of store-family level?
