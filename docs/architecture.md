# Architecture

```text
customers.csv  transactions.csv (1..n files)  prediction_requests.csv   [+ source_metadata.json]
        \                |                          /
         v               v                         v
   INGEST  - raw strings, source_file, source_row_number, schema-version check, required columns
        |
   VALIDATE - row rules, all reasons, duplicate policy ----> QUARANTINE (rejected_records + CSV)
        |  valid frames (typed, UTC)
        v
   FEATURES  build_features()  <- the single transformation (training == inference)
   LABELS    build_labels()    <- separate; training requests only; watermark-aware
        |
   PUBLISH   one SQLite transaction: upsert on logical keys, inserted/updated/unchanged counts
        |
   RECONCILE accounting, key integrity (sql/reconciliation.sql), independent leakage audit, parity
        |  (any failed check => run failed, nothing exported)
   EXPORT    atomic CSV replace: features, inference_features, labels, training_dataset, rejects
        |
   RUN LEDGER + REPORT (pipeline_runs row, outputs/reports/<run_id>.txt|json)
```

## Modules
| Module | Responsibility |
|---|---|
| `config.py`, `config/feature_config.json` | All window/boundary/vocabulary constants; validated; `config_hash` |
| `schemas.py` | Column contracts, vocabularies, feature/label column order |
| `generate_data.py` | Seeded synthetic data incl. boundary, late, invalid fixtures |
| `ingest.py` | Reading + schema-level failures |
| `validate.py` | Row-level validation, duplicate policy, quarantine frames |
| `features.py` | Optimized builder + naive reference builder |
| `labels.py` | Labels + training-dataset assembly |
| `publish.py` | DDL, upserts, atomic publish, run ledger |
| `reconcile.py` + `sql/reconciliation.sql` | Post-publish checks |
| `metrics.py` | Run id, stage timing, failed-stage capture |
| `pipeline.py` | Orchestration + CLI |
| `benchmark.py`, `demo.py` | Measurement and leakage walkthrough |

## Failure semantics
* Schema failure (missing file/column, unsupported `schema_version`): run fails at `ingest`.
* Any stage exception: ledger row `status=failed`, `failed_stage`, `error_message`; exception re-raised; exit code 1.
* Publish failure: the SQLite transaction rolls back, so no table is partially updated.
* Reconciliation failure: run is `failed` and **no output files are written/replaced**.
* Retry = run again (new `run_id`). Upserts on logical keys make this safe.
