# Walmart Foods Demand Forecasting

## Project Overview

This project analyzes historical Walmart sales data and develops a machine learning model to predict daily food demand. The project uses Python, pandas, and scikit-learn to prepare data, explore sales patterns, evaluate forecasting models, and generate demand predictions.

The main objective is to demonstrate how historical sales information can support demand planning and inventory decisions.

## Objectives

* Examine historical sales trends and patterns.
* Assess data quality and investigate potential outliers.
* Prepare a daily food-demand time series.
* Develop a Seasonal Naive baseline forecasting model.
* Train and evaluate a Random Forest regression model.
* Compare forecasting performance using MAE, RMSE, and WAPE.
* Save the trained model and generate operational prediction outputs.
* Propose a deployment architecture for future production use.

## Dataset

The project uses the Kaggle M5 Forecasting Accuracy dataset.

The original sales dataset contains 30,490 product-store records and daily sales columns. The data was aggregated to create a daily food-demand time series containing 1,913 observations.

The aggregated series covers January 29, 2011, through April 24, 2016.

The current forecasting model uses the aggregated `date` and `units_sold` fields.

## Methodology

The project follows these stages:

1. Dataset inspection and data-quality assessment.
2. Exploratory data analysis and visualization.
3. Outlier analysis.
4. Daily food-demand data preparation.
5. Feature engineering using calendar variables, historical sales lags, and rolling averages.
6. Chronological training, validation, and test splitting.
7. Seasonal Naive baseline development.
8. Random Forest model training and evaluation.
9. Model comparison and feature-importance analysis.
10. Saving the trained model and generating predictions.

The model uses historical features including lagged sales, rolling means, and calendar information. Historical observations are used to prevent future information from leaking into model features.

## Model Configuration

The selected Random Forest configuration is:

* `n_estimators=300`
* `max_depth=12`
* `min_samples_leaf=2`
* `random_state=42`
* `n_jobs=-1`

The baseline forecast uses sales from the corresponding day of the previous week.

## Evaluation Results

### Validation Results

| Metric | Seasonal Naive | Random Forest |
| ------ | -------------: | ------------: |
| MAE    |       2,590.36 |      1,573.06 |
| RMSE   |       3,146.37 |      2,284.10 |
| WAPE   |          9.17% |         5.57% |

### Test Results

| Metric | Seasonal Naive | Random Forest |
| ------ | -------------: | ------------: |
| MAE    |       2,440.93 |      1,097.72 |
| RMSE   |       2,992.56 |      1,412.41 |
| WAPE   |          8.64% |         3.89% |

On the historical test set, the Random Forest reduced MAE by approximately 55.03% and RMSE by approximately 52.80% compared with the Seasonal Naive baseline.

These results describe performance on the available historical test period. They do not guarantee the same accuracy on future data.

## Project Structure

```text
Walmart_Demand_Forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── figures/
├── models/
├── results/
├── src/
│   ├── data_quality_assessment.py
│   ├── exploratory_data_analysis.py
│   ├── inspect_dataset.py
│   ├── outlier_analysis.py
│   ├── prepare_forecasting_data.py
│   ├── run_forecast.py
│   └── train_forecasting_models.py
├── main.py
├── README.md
└── requirements.txt
```

## How to Run the Project

### 1. Create and activate the virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Generate operational predictions

From the project root, run:

```powershell
python src/run_forecast.py
```

The script loads the processed daily-demand dataset and saved Random Forest model, validates the data, creates the forecasting features, and saves the prediction results.

## Output Files

The project produces analytical results, visualizations, model evaluation metrics, and prediction files.

Important files include:

* `models/walmart_food_random_forest.joblib` — saved Random Forest model.
* `results/forecasting_model_metrics.csv` — model evaluation metrics.
* `results/feature_importance.csv` — model feature-importance results.
* `results/validation_predictions.csv` — validation predictions.
* `results/test_predictions.csv` — test predictions.
* `results/operational_forecast.csv` — operational script output containing historical predictions.

## Deployment Status

The forecasting pipeline has been executed locally, and its operational CSV output has been verified.

The current operational script generates historical one-day-ahead predictions using actual preceding sales. It does not yet generate genuine future forecasts beyond April 24, 2016.

Cloud hosting, automated scheduling, production monitoring, and a future-date forecasting workflow remain proposed improvements. They should not be interpreted as completed deployments.

## Limitations and Future Work

Future improvements include:

* Implementing and evaluating forecasts for dates beyond the available historical data.
* Adding automated tests and structured logging.
* Scheduling forecasting runs.
* Monitoring data quality and forecasting performance.
* Evaluating the model on newer data.
* Deploying the application to a cloud environment after production-readiness testing.

## Conclusion

The project demonstrates a complete local workflow for historical food-demand analysis, machine learning model development, evaluation, and operational prediction generation. The Random Forest outperformed the Seasonal Naive baseline on the historical test set. The results provide a foundation for further development into an automated demand-forecasting application.
