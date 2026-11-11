# Chronos design contract (v1)

## Question answered
At prediction timestamp `T`, which customer-activity features can safely be given
to a model predicting a qualifying purchase in the next 7 days?

## Table grain and keys
| Table | Grain | Key |
|---|---|---|
| customers | one row per customer | `customer_id` |
| transactions | one row per event | `event_id` |
| prediction_requests | one row per customer + timestamp | `prediction_id` (unique); `customer_id + prediction_timestamp` natural key |
| customer_features | one row per prediction + feature version | `(prediction_id, feature_version)` |
| training_labels | one row per prediction + label version | `(prediction_id, label_version)` |
| pipeline_runs | one row per execution | `run_id` |
| rejected_records | one row per rejected source row | `(run_id, source_file, source_row_number)` |

## Two clocks
- `event_timestamp`: when it happened. `available_at`: when the pipeline could know.
- Synthetic source rule: `available_at >= event_timestamp`.

## Feature window (all UTC, tz-aware)
An event is eligible for prediction `T` iff:
`T - 30d < event_timestamp <= T` AND `available_at <= T` AND same `customer_id`.
Lower bound exclusive, upper bound inclusive.

## Label window
Qualifying purchase: `T < event_timestamp <= T + 7d`, type `purchase`, status `completed`.
Label is finalised only if `data_complete_through >= T + 7d`; otherwise
`label_status = incomplete` and target is NULL (never 0).

## Qualifying purchase / refunds
Only `purchase` + `completed` counts. Refunds are counted separately, never netted in v1.
Single currency (USD) in v1; amounts never summed across currencies.

## Duplicate policy
- same `event_id`, same content -> replay; one canonical event
- same `event_id`, different content -> conflict; quarantine
- similar customer/time/amount with different IDs -> NOT deduplicated

## Schema-level vs row-level failures
- Missing required column / unsupported schema version -> file-level failure, clear error.
- Malformed timestamp, null ID, bad vocab, bad amount, unknown customer -> row-level reject
  with source file, row number, raw record and reason(s).

## Null policy
count/sum -> 0 when no history; avg and days_since_last_purchase -> NULL when no purchases.
Prediction before signup -> rejected request.
