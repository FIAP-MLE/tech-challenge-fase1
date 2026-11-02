# FIAP Machine Learning Engineering - Telco Churn

This repository contains the FIAP Machine Learning Engineering Tech Challenge
project for exploratory analysis, churn model training, experiment tracking,
and a FastAPI prediction service.

Repository: <https://github.com/FIAP-MLE/tech-challenge-fase1>

## Project layout

- `data/` - source workbook and generated encoded data, versioned with DVC.
- `data.dvc` and `models.dvc` - DVC metadata for the data and model artifacts.
- `notebooks/step1-exploratory-data-analysis.ipynb` - EDA and preprocessing.
- `notebooks/step2-ml-modeling-and-evaluation.ipynb` - model comparison and
  champion selection.
- `models/` - DVC-managed model artifacts.
- `src/` - FastAPI application and raw-to-encoded request translation.
- `tests/` - API and preprocessing regression tests.
- `docs/results/` - experiment results and model comparison.

## Installation

```bash
git clone https://github.com/FIAP-MLE/tech-challenge-fase1.git
cd tech-challenge-fase1

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The source workbook and generated encoded data are included in the configured
DVC remote, together with the trained model artifacts. After installing DVC,
restore all versioned data and models with:

```bash
dvc pull data.dvc models.dvc
```

This restores `data/Telco_customer_churn.xlsx`,
`data/telco_full_encoded.csv`, and the files under `models/`. DVC credentials
may be required for private remote access. The API requires
`models/telco_random_forest_champion.joblib`.

## Running the notebooks

From the repository root:

```bash
jupyter notebook
```

Open and execute the notebooks in order. The first notebook creates the
encoded dataset used by the second notebook. Both notebooks also contain
project-local MLflow setup so they can be run from a PyCharm notebook
configuration or from the `notebooks/` working directory.

## Running the API from the command line

Run Uvicorn from the repository root:

```bash
source .venv/bin/activate
uvicorn src.main:app --reload
```

The interactive documentation is available at
<http://127.0.0.1:8000/docs>. Health and metadata endpoints are available at
`/health` and `/`.

The prediction endpoint is `POST /predict`. It accepts the original dataset
fields; callers do not need to know about one-hot or frequency encoding. For
example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "City": "Los Angeles",
    "Zip Code": 90001,
    "Latitude": 33.9736,
    "Longitude": -118.2490,
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 75.5,
    "Total Charges": 906.0,
    "CLTV": 5000
  }'
```

`Churn Score` is not requested because it is a derived, post-evaluation field
and is unavailable when scoring a current customer.

## Running from PyCharm

1. Open the cloned repository as a project.
2. Configure a Python interpreter using the repository `.venv`.
3. Install dependencies from `requirements.txt`.
4. Create a Python run configuration with module name `uvicorn` and parameters
   `src.main:app --reload`; set the working directory to the repository root.
5. For notebooks, configure the notebook working directory as either the
   repository root or `notebooks/`.

## Tests

Run the regression suite from the repository root:

```bash
pytest
```

The tests cover the public routes, request/response validation, raw feature
encoding, unknown-city validation, and model artifact loading.

## Champion model card

### Model details

- **Model:** `RandomForestClassifier`
- **Artifact:** `models/telco_random_forest_champion.joblib`
- **Task:** binary classification of customer churn
- **Selection:** champion chosen against a scaled `MLPClassifier` using
  stratified five-fold cross-validation with F1 as the primary metric
- **Random state:** 42
- **Parameters:** 300 trees, `min_samples_leaf=2`,
  `class_weight="balanced"`, and `n_jobs=-1`

### Intended use

The model is intended to support exploratory customer-retention workflows by
estimating whether an individual Telco customer is likely to churn. It can
help prioritize human review and retention outreach. It must not be used as
the sole basis for denying service, changing prices, or making decisions that
materially affect a customer.

### Data and features

Training data comes from the Telco customer churn workbook used in the
challenge. The API accepts 24 raw customer and service fields. Preprocessing
converts `Total Charges` to numeric form, applies one-hot encoding to
categorical fields, and frequency-encodes `City`.

`Churn Score`, `Churn Label`, `Churn Reason`, and other target-derived fields
are excluded to prevent target leakage and because they are not available at
prediction time. The model uses 35 encoded predictors, including numeric
customer, billing, geographic, service, and encoded city-frequency features.

### Evaluation

The data is split into a stratified 80/20 train/test partition, with
five-fold stratified cross-validation applied to the training partition.
F1 is the primary metric because the target is imbalanced and both missed
churners and unnecessary interventions matter.

The latest recorded results are:

| Metric | Random Forest |
|---|---:|
| Cross-validation F1 mean | 0.6466 |
| Cross-validation F1 standard deviation | 0.0263 |
| Test F1 | 0.6315 |
| Test ROC AUC | 0.8465 |
| Test balanced accuracy | 0.7587 |
| Test precision | 0.5628 |
| Test recall | 0.7193 |

The Random Forest was selected over the MLP because it achieved the stronger
cross-validated F1 and held-out F1 under the predefined selection rule.

### Limitations and risks

- The dataset is a historical sample and may not represent every current
  customer population or future service configuration.
- Predictions are probabilistic estimates, not explanations or guarantees.
- Geographic and service variables can encode socioeconomic or regional
  patterns that require monitoring for disparate performance.
- The city-frequency mapping is learned from the training workbook; cities
  absent from that workbook are rejected by the current API.
- Model performance can drift as pricing, contracts, products, and customer
  behavior change.
- The current API is a single-record scoring service and does not provide
  threshold optimization, monitoring dashboards, authentication, or audit
  storage.

### Operational guidance

Keep the model artifact and preprocessing logic together: changing feature
encoding requires retraining and replacing the champion artifact as a unit.
Validate inputs through the OpenAPI schema, monitor prediction distributions
and outcome quality over time, and retrain when data or business conditions
change. Store credentials outside the repository and keep MLflow state local
or in an appropriately secured tracking service.
