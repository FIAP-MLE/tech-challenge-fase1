# Phase 2 Model Comparison Results

## Scope

This experiment compares a Random Forest and a scaled scikit-learn `MLPClassifier` for Telco customer churn. Both models use the same encoded numeric features and the same stratified 80/20 train/test split (random state 42). Cross-validation is applied only to the training partition using five shuffled stratified folds.

## Evaluation criteria

F1 score is the primary selection metric because the churn target is imbalanced and both precision and recall matter. ROC AUC, balanced accuracy, precision, recall, accuracy, and fold-to-fold standard deviation are supporting metrics.

- Training rows: 5634
- Test rows: 1409
- Numeric features: 36
- Cross-validation: StratifiedKFold, 5 folds, shuffle=True, random_state=42
- Feature exclusions: CustomerID, Count, Country, State, target

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
| Random Forest | 0.8720 | 0.0216 | 0.9770 | 0.8571 | 0.9769 | 0.9218 | 0.7945 | 0.9305 |
| MLPClassifier | 0.8322 | 0.0128 | 0.9705 | 0.8331 | 0.9723 | 0.8834 | 0.8457 | 0.8209 |

## Champion selection

**Selected model: Random Forest.** It achieved the strongest cross-validated F1 under the predefined selection rule, with ROC AUC and held-out F1 used as tie-breakers. The serialized champion is `models/telco_random_forest_champion.joblib`.

The two non-champion model artifacts are also persisted under `models/` for reproducibility and future comparison.
