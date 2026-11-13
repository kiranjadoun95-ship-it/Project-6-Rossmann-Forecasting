
# Project 6: Sales Forecasting Across Multiple Retail Stores

## Notebook 04: Machine Learning Modeling and Evaluation

### 1. Selected model

- Selected model: RF_More_Trees
- Model version: RandomForest_05-11-2026-17-11-55-00
- Model stage: Tuned pilot, not full-training production model
- Training rows used: 60,000
- Available full training rows: 970,379
- Validation rows: 46,830
- Processed features: 1,149

### 2. Selected hyperparameters

```json
{
    "n_estimators": 60,
    "max_depth": 18,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "bootstrap": true,
    "max_samples": 0.8,
    "criterion": "squared_error",
    "random_state": 42,
    "n_jobs": 2
}
```

### 3. Validation performance

| Model | Model_Status | Training_Rows | Validation_Rows | MAE | RMSE | RMSPE_Percent |
| --- | --- | --- | --- | --- | --- | --- |
| Dummy - Full Training | Baseline | 970379 | 46830 | 2757.65 | 3733.83 | 45.78 |
| Dummy - Pilot Sample | Matched baseline | 60000 | 46830 | 2758.12 | 3734.57 | 45.69 |
| RF_More_Trees | Selected tuned pilot | 60000 | 46830 | 1898.08 | 2570.83 | 42.39 |

### 4. Relative error reduction against pilot Dummy

| Metric | Pilot_Dummy | Selected_RF | Relative_Error_Reduction_Percent |
| --- | --- | --- | --- |
| MAE | 2758.12 | 1898.08 | 31.18 |
| RMSE | 3734.57 | 2570.83 | 31.16 |
| RMSPE_Percent | 45.69 | 42.39 | 7.23 |

RMSPE reduction: 3.30 percentage points.

### 5. Top original features

| Rank | Original_Feature | Importance_Percent |
| --- | --- | --- |
| 1 | Open | 23.309 |
| 2 | DayOfWeek | 21.078 |
| 3 | IsWeekend | 12.466 |
| 4 | Promo | 11.751 |
| 5 | Store | 11.325 |
| 6 | StateHoliday | 9.033 |
| 7 | DaysUntilHoliday | 1.88 |
| 8 | StoreType | 1.662 |
| 9 | Assortment | 1.114 |
| 10 | WeekOfYear | 1.058 |

Impurity-based feature importance indicates model reliance; it does not establish causation.

### 6. Prediction interval assessment

- Nominal interval coverage: 90.00%
- Actual overall validation coverage: 63.42%
- Coverage gap: 26.58 percentage points.

The empirical tree-spread intervals are not calibrated or guaranteed 90% prediction intervals.

### 7. Model serialization

- Model file: `D:\Project_6_Rossmann_Forecasting\models\machine_learning\RandomForest_05-11-2026-17-11-55-00.pkl`
- Metadata file: `D:\Project_6_Rossmann_Forecasting\models\machine_learning\RandomForest_05-11-2026-17-11-55-00_metadata.json`
- Registry file: `D:\Project_6_Rossmann_Forecasting\models\machine_learning\model_version_registry.csv`
- Reloaded predictions matched: True

### 8. Source metadata audit

- Original Notebook 03 metadata audit passed: True
- Recorded discrepancies: []

The original metadata was preserved; discrepancies are disclosed rather than silently overwritten.

### 9. Limitations and pending work

- The selected Random Forest is a tuned pilot trained on 60,000 rows, not the complete 970,379-row training dataset.
- The chronological validation set was used for hyperparameter selection; validation metrics are not an independent final generalization estimate.
- The empirical tree-spread interval achieved 63.42% overall validation coverage against a nominal 90.00% level; it is not calibrated to the nominal coverage.
- Tree-to-tree prediction spread is not a guaranteed statistical confidence interval or prediction coverage guarantee.
- The original Notebook 03 metadata JSON has recorded discrepancies: missing train_rows and preprocessing_fitted_on fields. The source metadata was preserved.
- This model predicts Sales only. A separate Customers prediction model is still required for the dashboard requirement.
- LSTM, MLflow serving and the web dashboard remain separate project stages.

### 10. Conclusion

The selected Random Forest pilot was chosen using validation RMSPE among the evaluated configurations. It improved validation errors against the matched pilot Dummy baseline, but it remains a 60,000-row pilot and requires further final-stage evaluation.

LSTM modelling, MLflow, a separate Customers model, and the prediction dashboard remain pending.
