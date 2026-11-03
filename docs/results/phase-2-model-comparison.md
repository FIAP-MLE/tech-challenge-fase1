# Model Comparison Results

## Scope

This experiment compares the Step 1 Logistic Regression MVP baseline with a Random Forest and a scaled scikit-learn `MLPClassifier`. All three models use the same encoded numeric features, stratified 80/20 split, and five-fold stratified cross-validation on the training partition.

## Evaluation criteria

F1 score is the primary selection metric because the churn target is imbalanced and both precision and recall matter. CV ROC AUC and held-out test F1 are tie-breakers; balanced accuracy, precision, recall, accuracy, and confusion matrices provide additional context.

- Training rows: 5634
- Test rows: 1409
- Prediction-time numeric features: 35
- Cross-validation: StratifiedKFold, 5 folds, shuffle=True, random_state=42
- Feature exclusions: CustomerID, Count, Country, State, target (derived churn fields are removed during Step 1 preprocessing)

## Model parameters

### Logistic Regression (MVP baseline)

```python
LogisticRegression(random_state=42)
```

### Random Forest

```python
RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
```

### MLPClassifier

```python
Pipeline([StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0005, batch_size=64, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)])
```

## Results

| Model | Role | CV F1 mean | CV F1 std | CV ROC AUC mean | Test F1 | Test ROC AUC | Test balanced accuracy | Test precision | Test recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (MVP baseline) | Baseline | 0.5367 | 0.0108 | 0.8146 | 0.5659 | 0.8309 | 0.7022 | 0.6346 | 0.5107 |
| Random Forest | Candidate | 0.6466 | 0.0263 | 0.8575 | 0.6315 | 0.8465 | 0.7587 | 0.5628 | 0.7193 |
| MLPClassifier | Candidate | 0.6095 | 0.0398 | 0.8493 | 0.5616 | 0.8411 | 0.6993 | 0.6404 | 0.5000 |

## Champion selection

**Selected model: Random Forest.** The baseline and both candidate models were evaluated with the same data split and validation protocol. The champion is selected first by cross-validated F1, then by CV ROC AUC and held-out test F1. The serialized champion is `models/telco_random_forest_champion.joblib`.

The Logistic Regression MVP baseline is retained as `models/telco_full_logistic_regression_model.joblib` for reproducibility. The non-champion candidate is also persisted under `models/`.
