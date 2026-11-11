# Chronos — Point-in-Time Correct ML Feature Engineering Pipeline

Chronos is a reproducible batch pipeline that builds historical customer features for a seven-day purchase prediction use case. Its central requirement is **point-in-time correctness**: a feature row may use only information that was both generated and available by that row's prediction timestamp.

This repository demonstrates data ingestion, schema and business validation, rejected-record traceability, feature and label generation, SQLite publishing, reconciliation, idempotent reruns, automated tests, and measured performance. It is a local, production-minded project demonstration—not a deployed production service.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd chronos-ml-feature-pipeline
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
python -m src.pipeline
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard provides:
- Pipeline execution and status monitoring
- Feature Explorer for generated feature datasets
- Generated target labels
- Data quality and reconciliation checks
- Rejected record inspection
- Pipeline execution metrics and stage timings

## 1. Problem and scope

The pipeline answers:

> At prediction timestamp `T`, what customer activity was known during the previous 30 days, and did the customer make a qualifying purchase during the following seven days?

| Item | Contract |
|---|---|
| Entity | Customer |
| Prediction grain | One row per `prediction_id` |
| Feature lookback | 30 days before `T` |
| Label horizon | 7 days after `T` |
| Time standard | UTC |
| Qualifying purchase | `event_type = purchase` and `event_status = completed` |
| Feature output | Historical features only; no target column |
| Label output | Separate future-outcome artifact/table |
| Inference output | Features only |

The project does not train or deploy a predictive model. The generated data is synthetic and is used to test pipeline behavior, not to make claims about real customers.

## 2. Point-in-time correctness

For a prediction timestamp `T`, an event is eligible for a 30-day feature window only when all conditions hold:

```text
T - 30 days < event_timestamp <= T
available_at <= T
customer_id matches the prediction request
```

The lower lookback boundary is exclusive; the prediction-time boundary is inclusive. `event_timestamp` records when the activity occurred. `available_at` records when the pipeline could know about it. Both are required to prevent temporal and late-arrival leakage.

Labels use a separate future window:

```text
T < event_timestamp <= T + 7 days
```

A label is final only when the source completeness watermark covers the full seven-day horizon. If the horizon is not fully observable, the target remains null and the label is marked incomplete; it is not incorrectly converted to a negative outcome.

The same feature-building logic is used for training and inference. The training dataset joins complete labels to historical features; labels are not included in the inference feature artifact.

## 3. Architecture

```text
Customers / transactions / prediction requests (CSV)
                         |
                         v
                      INGEST
                         |
                         v
                SCHEMA + DATA VALIDATION
                         |
               +---------+---------+
               |                   |
               v                   v
          Valid records       Rejected records
               |              + reason/traceability
               v
          Canonical events
               |
               v
      Point-in-time feature builder
               |
        +------+------+
        |             |
        v             v
  Feature outputs  Future label builder
        |             |
        |        Complete/incomplete labels
        |             |
        +------+------+
               v
        SQLite publishing
               |
               v
     Reconciliation + run report
               |
               v
       CSV exports / tests
```

### Technology

- **Python** for pipeline logic and orchestration.
- **Pandas** for batch data processing.
- **SQLite** for local persistence and transactional publishing.
- **CSV** for transparent input and output artifacts.
- **pytest** for automated correctness and edge-case tests.
- Python logging and JSON configuration for run diagnostics and reproducibility.

## 4. Feature contract

The versioned feature output contains one row per valid prediction request and the following columns:

| Feature | Meaning |
|---|---|
| `prediction_id` | Prediction request identifier |
| `customer_id` | Customer identifier |
| `prediction_timestamp` | Point-in-time cutoff `T` |
| `feature_version` | Feature definition version |
| `purchase_count_30d` | Count of qualifying purchases in the eligible lookback |
| `purchase_amount_sum_30d` | Sum of qualifying purchase amounts |
| `purchase_amount_avg_30d` | Average qualifying purchase amount |
| `days_since_last_purchase` | Days since the most recent eligible qualifying purchase |
| `refund_count_30d` | Eligible refund count under the implemented event rules |
| `login_count_30d` | Eligible login count |
| `customer_tenure_days` | Customer tenure at prediction time |

Feature definitions and source schemas are documented in `docs/data_dictionary.md`, `docs/design_contract.md`, and `config/feature_config.json`.

## 5. Repository layout

```text
config/                 Feature configuration
 data/                  Sample data and generated-data location
 docs/                  Architecture, contracts, trade-offs, evidence
   evidence/            Captured run, test, demo, and benchmark outputs
 outputs/                Generated CSV exports, reports, and SQLite databases
 sql/                    Reconciliation SQL
 src/                    Pipeline implementation
 tests/                  Automated tests
 README.md
 requirements.txt
 .gitignore
```

Main implementation modules include ingestion, validation, feature generation, label generation, publishing, reconciliation, metrics, benchmarking, demo walkthrough, and synthetic data generation.

## 6. Environment and setup

**Verified environment:** Windows, Python 3.14.3, pandas 3.1.1, pytest 9.2.0.

From the repository root, in Windows PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the Python executable inside `venv\Scripts\python.exe` directly or adjust the execution policy for the current session according to your organization's policy.

On macOS/Linux, create and activate a virtual environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. Run the project

### Default sample pipeline

```powershell
py -m src.pipeline
```

The default run uses the committed sample data under `data/sample` and writes outputs under `outputs/`.

### Run the demo walkthrough

```powershell
py -m src.demo
```

The demo illustrates point-in-time behavior, including the effect of late-arriving and future events.

### Generate a larger deterministic dataset

```powershell
py -m src.generate_data --customers 5000 --events 100000 --seed 42 --out-dir data/generated
```

The generator creates 5,000 customers, 100,000 requested events plus 120 boundary events, and 20,000 prediction requests. It also writes source metadata.

Run the generated dataset in a separate database and output directory:

```powershell
py -m src.pipeline `
  --customers data/generated/customers.csv `
  --transactions data/generated/transactions.csv `
  --predictions data/generated/prediction_requests.csv `
  --db outputs/generated.db `
  --out-dir outputs/generated
```

### Exercise validation and quarantine

```powershell
py -m src.pipeline `
  --transactions data/sample/transactions.csv data/sample/invalid_transactions.csv `
  --db outputs/invalid_demo.db `
  --out-dir outputs/invalid_demo
```

Use a separate database/output directory for this negative-path demonstration so the default clean database remains unchanged.

### Run automated tests and benchmark

```powershell
py -m pytest -v
py -m src.benchmark
```

## 8. Outputs

The default run produces a SQLite database and CSV/report artifacts under `outputs/`. Key artifacts include:

| Artifact | Purpose |
|---|---|
| `outputs/chronos.db` | SQLite database for the default run |
| `outputs/features/features_v1.csv` | Historical feature rows; no target column |
| `outputs/features/inference_features_v1.csv` | Inference-only feature rows |
| `outputs/features/training_dataset_v1.csv` | Features joined to complete labels |
| `outputs/labels/labels_v1.csv` | Separate complete and incomplete labels |
| `outputs/reports/` | Run and reconciliation reports |
| `outputs/rejects/` | Rejected records with reason and source traceability |

The generated-data and invalid-data commands use their explicitly supplied output directories and database paths.

## 9. Validation, rejection, and accounting

The pipeline validates source structure and business rules, including required identifiers, customer references, event types/statuses, timestamps, amount/currency rules, and duplicate event identifiers. Invalid records are written to a reject path with a reason and source traceability. Replayed duplicate records are accounted for separately from rejected records; conflicting duplicate identifiers are treated as invalid.

For each source, the accounting equation is:

```text
input = valid + rejected + duplicate
```

The pipeline also reconciles feature and label output counts against valid prediction requests, checks keys and customer relationships, checks label watermark rules, and runs leakage/parity audits.

## 10. Recorded execution evidence

The following results were captured from the project runs described below. Runtime measurements are machine- and environment-dependent.

### Clean sample run — Windows

Environment: Python 3.14.3, pandas 3.1.1, pytest 9.2.0.

| Metric | Result |
|---|---:|
| Run status | Success |
| Input events | 3,120 |
| Valid events | 3,120 |
| Rejected / duplicate events | 0 / 0 |
| Prediction requests | 600 (600 valid) |
| Feature rows | 600 |
| Complete label rows | 412 |
| Incomplete label rows | 71 |
| Feature version | v1 |
| Data completeness watermark | `2026-10-01T00:00:00Z` |
| Run duration | 1.3497 seconds |

The run's source accounting and reconciliation checks passed. The label output contains 483 records in total: 412 complete and 71 incomplete. The training dataset uses the 412 complete labels; the remaining requests without labels remain inference-only or have incomplete label coverage according to the pipeline's output contract.

### Larger generated-data run — Windows

| Metric | Result |
|---|---:|
| Customers | 5,000 |
| Input events | 100,120 |
| Valid / rejected / duplicate events | 100,120 / 0 / 0 |
| Prediction requests | 20,000 |
| Feature rows | 20,000 |
| Complete labels | 13,900 |
| Incomplete labels | 2,047 |
| Requests without a label record | 4,053 |
| Run duration | 10.1281 seconds |
| Reconciliation | All checks passed |

The 4,053 requests without a label record are not counted as complete or incomplete labels in the reported label output. The generated run's completeness watermark was `2026-10-01T00:00:00Z`.

### Invalid-data demonstration

A separate run included invalid fixtures and a replay duplicate:

- Input events: 3,135
- Valid events: 3,119
- Rejected events: 15
- Duplicate events: 1
- Rejected-record rows published: 16 (including the duplicate record's traceability)
- Reconciliation: all checks passed

The reject output includes examples such as null identifiers, unknown customers, invalid event types/statuses, invalid timestamps, missing or invalid amounts, invalid currency, availability before event occurrence, replay duplicates, and conflicting duplicate event identifiers.

### Idempotent rerun

The same sample inputs were run again against the same database. The second run created a new run record but did not insert or update logical data rows:

| Table | Inserted | Updated | Unchanged |
|---|---:|---:|---:|
| customers | 0 | 0 | 100 |
| canonical_events | 0 | 0 | 3,120 |
| prediction_requests | 0 | 0 | 600 |
| customer_features | 0 | 0 | 600 |
| training_labels | 0 | 0 | 483 |
| rejected_records | 0 | 0 | 0 |
| **Total** | **0** | **0** | **3,883** |

This demonstrates idempotency for identical inputs under the tested publishing path. It does not claim distributed exactly-once processing.

## 11. Automated tests

Latest recorded Windows test run:

```text
86 passed in 8.12s
Python 3.14.3
pandas 3.1.1
pytest 9.2.0
```

The test suite covers valid and empty inputs, missing columns, unsupported schema, malformed and naive timestamps, null identifiers, unknown customers, invalid event vocabularies and amounts, currency validation, replay and conflicting duplicates, point-in-time window boundaries, late arrivals, other-customer isolation, appended future events, positive/negative/incomplete labels, label boundaries, no-history behavior, training/inference parity, idempotent reruns, publish rollback and retry, ingest failure, reconciliation failure, deterministic generation, and optimized/reference output equality.

The leakage tests were mutation-checked: deliberately changing the lower-bound rule, ignoring `available_at`, or changing the upper-bound rule caused the relevant tests to fail.

## 12. Performance benchmark

Latest recorded benchmark run: Windows, Python 3.14.3, pandas 3.1.1, seed 42. Event counts include 120 boundary events. These timings are measurements from this machine and should not be treated as portable service-level guarantees.

| Events | Predictions | Ingest (s) | Validate (s) | Features (s) | Publish (s) | Reconcile (s) | Total Runtime (s) | Events/sec |
|-------:|------------:|-----------:|-------------:|-------------:|------------:|--------------:|------------------:|-----------:|
| 1,120 | 200 | 0.0748 | 0.0631 | 0.0118 | 0.0484 | 0.3302 | 0.7548 | 1,483.8 |
| 10,120 | 2,000 | 0.0862 | 0.1446 | 0.0687 | 0.2363 | 0.7053 | 1.6829 | 6,013.4 |
| 100,120 | 20,000 | 0.2000 | 1.3268 | 0.8607 | 2.0919 | 3.3132 | 11.4185 | 8,768.2 |

### Feature-builder comparison

The reference implementation filters the full transaction table for each prediction request. The optimized implementation sorts/group-indexes the data and narrows the search to the relevant customer's time window before applying availability-time filtering. For the compared request samples, outputs were identical.

| Prediction Requests | Reference Implementation (s) | Optimized Implementation (s) | Speedup |
|--------------------:|-----------------------------:|-----------------------------:|--------:|
| 200 | 0.2901 | 0.0113 | 25.6x |
| 2,000 | 3.9919 | 0.0705 | 56.6x |
| 5,000 | 46.3239 | 0.3569 | 129.8x |

Note: The reference implementation was benchmarked against the optimized feature builder for up to 5,000 prediction requests. The optimized implementation was also evaluated as part of the full pipeline with 20,000 prediction requests. Benchmark timings are environment-dependent and may vary across machines.

At the largest benchmark size, the reference implementation was measured on the first 5,000 requests only; the optimized pipeline processed all 20,000 prediction requests in the full generated-data run. The 5,000-request comparison is an apples-to-apples feature-builder comparison, not a claim that the reference implementation processed all 20,000 requests. The benchmark's total runtime (9.7365 seconds) is a separate run from the full generated-data pipeline run (10.1281 seconds).

## 13. Idempotency, failure handling, and backfills

Publishing uses a transaction and logical keys/upserts so identical reruns converge without creating duplicate logical rows. The project includes failure-path tests for ingestion and publishing rollback/retry behavior. Run records capture execution status and failure stage.

For a historical backfill, provide prediction requests with historical timestamps alongside the corresponding event snapshot. The same point-in-time eligibility rules apply. When a feature definition changes, increment the feature version rather than silently changing the meaning of existing feature rows.

## 14. Known limitations

- Batch-snapshot processing; this is not a streaming or incremental pipeline.
- Local Pandas processing and SQLite persistence; no distributed compute or cloud deployment is included.
- The demonstration uses a single currency and static customer attributes.
- Feature names and definitions are tied to the documented 30-day lookback and seven-day label horizon.
- Synthetic data is used; real-world source-system semantics and operational SLAs would require separate validation.
- Reconciliation occurs after publishing in the current workflow.
- Memory usage was not measured as part of the recorded benchmark.

## 15. Documentation and evidence

Additional project documentation:

- `docs/architecture.md` — architecture and data flow.
- `docs/data_dictionary.md` — schemas and feature definitions.
- `docs/design_contract.md` — behavioral and data contracts.
- `docs/design_tradeoffs.md` — implementation choices and limitations.
- `docs/evidence/` — captured test, run, demo, reject, and benchmark evidence.

To reproduce the main checks from the repository root:

```powershell
py -m pytest -v
py -m src.pipeline
py -m src.demo
py -m src.benchmark
```

---

**Project summary:** Chronos demonstrates a tested, point-in-time-correct batch feature pipeline with separate future labels, traceable data-quality handling, reproducible outputs, idempotent publishing for identical inputs, reconciliation, and measured feature-generation optimization.
