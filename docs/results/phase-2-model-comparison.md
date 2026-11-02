# Phase 2 Model Comparison Results

## Scope

This experiment compares a Random Forest and a scaled scikit-learn `MLPClassifier` for Telco customer churn. Both models use the same encoded numeric features and the same stratified 80/20 train/test split (random state 42). Cross-validation is applied only to the training partition using five shuffled stratified folds.

## Evaluation criteria

F1 score is the primary selection metric because the churn target is imbalanced and both precision and recall matter. ROC AUC, balanced accuracy, precision, recall, accuracy, and fold-to-fold standard deviation are supporting metrics.

- Training rows: 5634
- Test rows: 1409
- Prediction-time numeric features: 35
- Cross-validation: StratifiedKFold, 5 folds, shuffle=True, random_state=42
- Feature exclusions: CustomerID, Count, Country, State, target (Churn Score is removed during Step 1 preprocessing)

## Model parameters

### Random Forest

```python
RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
```

### MLPClassifier

```python
Pipeline([StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', alpha=0.0005, batch_size=64, learning_rate_init=0.001, max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)])
```

## Results

| Model | CV F1 mean | CV F1 std | CV ROC AUC mean | Test F1 | Test ROC AUC | Test balanced accuracy | Test precision | Test recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6466 | 0.0263 | 0.8575 | 0.6315 | 0.8465 | 0.7587 | 0.5628 | 0.7193 |
| MLPClassifier | 0.6095 | 0.0398 | 0.8493 | 0.5616 | 0.8411 | 0.6993 | 0.6404 | 0.5000 |

## Champion selection

**Selected model: Random Forest.** It achieved the strongest cross-validated F1 under the predefined selection rule, with ROC AUC and held-out F1 used as tie-breakers. The serialized champion is `models/telco_random_forest_champion.joblib`.

The MVP and the non-champion model artifacts are also persisted under `models/` for reproducibility and future comparison.
