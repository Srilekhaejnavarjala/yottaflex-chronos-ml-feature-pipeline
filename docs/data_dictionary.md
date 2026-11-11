# Data dictionary (schema version 1)

All timestamps are timezone-aware **UTC**, ISO-8601 with `Z` (e.g. `2026-10-20T12:00:00Z`).
Timestamps without a timezone designator are rejected, never guessed.

## Source files

### customers.csv — grain: one row per customer — key: `customer_id`
| Column | Type | Rule |
|---|---|---|
| customer_id | string | required, unique |
| signup_timestamp | UTC timestamp | required |
| region, customer_segment | string | nullable; static in v1 |

### transactions.csv — grain: one row per event — key: `event_id`
| Column | Type | Rule |
|---|---|---|
| event_id | string | required, unique (see duplicate policy) |
| customer_id | string | required, must exist in valid customers |
| event_timestamp | UTC timestamp | when it happened |
| available_at | UTC timestamp | when the pipeline could know; must be `>= event_timestamp` |
| event_type | enum | `purchase, refund, login, cart_add, cancellation` |
| event_status | enum | `completed, pending, failed, cancelled` |
| amount | decimal | **required and > 0** for `purchase`/`refund`; must be blank for other types |
| currency | string | must equal `USD` for monetary events |

### prediction_requests.csv — grain: one row per customer + timestamp — key: `prediction_id`
| Column | Rule |
|---|---|
| prediction_id | required, unique |
| customer_id | must exist; natural key `(customer_id, prediction_timestamp)` must be unique |
| prediction_timestamp | UTC cutoff `T`; must be `>= signup_timestamp` |
| request_type | `training` or `inference` |

Extra columns in any file are ignored for processing and preserved in `raw_record` of rejects.
`source_row_number` is the 1-based file line (header = line 1).

## Published tables (SQLite)

| Table | Grain | Unique key |
|---|---|---|
| `customers` | customer | `customer_id` |
| `canonical_events` | valid event | `event_id` |
| `prediction_requests` | valid request | `prediction_id`, `(customer_id, prediction_timestamp)` |
| `customer_features` | request x feature version | `(prediction_id, feature_version)` |
| `training_labels` | training request x label version | `(prediction_id, label_version)` |
| `rejected_records` | rejected/duplicate source row per run | `(run_id, source_table, source_file, source_row_number, category)` |
| `pipeline_runs` | execution | `run_id` |

## Features (`feature_version = v1`)
Eligible event for request time `T`: `T - 30d < event_timestamp <= T` AND `available_at <= T` AND same customer.

| Feature | Definition | dtype | No-history value |
|---|---|---|---|
| purchase_count_30d | completed purchases | int | 0 |
| purchase_amount_sum_30d | sum of those amounts (rounded to 2 dp, USD only) | float | 0.0 |
| purchase_amount_avg_30d | mean of those amounts (4 dp) | float | NULL |
| days_since_last_purchase | `T - latest eligible purchase`, fractional days (4 dp) | float | NULL |
| refund_count_30d | completed refunds (never netted against purchases) | int | 0 |
| login_count_30d | completed logins | int | 0 |
| customer_tenure_days | whole days from signup to `T` | int | n/a (request before signup is rejected) |

Zero count and NULL average/last-purchase are intentionally different: *no purchases* vs *undefined*.

## Labels (`label_version = v1`) — stored separately
| Column | Meaning |
|---|---|
| target_purchase_next_7d | 1 if a completed purchase has `T < event_timestamp <= T + 7d`, else 0; **NULL if incomplete** |
| label_window_end | `T + 7d` |
| label_status | `complete` iff `data_complete_through >= T + 7d`, else `incomplete` |

Labels exist for `training` requests only. Features and labels meet only in `training_dataset_v1.csv`
(complete labels, joined by `prediction_id`).

## Reject reasons
`null_event_id, null_customer_id, invalid_event_timestamp, invalid_available_at, invalid_event_type,
invalid_event_status, unknown_customer, available_before_event, invalid_amount_format, missing_amount,
non_positive_amount, unexpected_amount, invalid_currency` (+ `null_prediction_id, invalid_prediction_timestamp,
invalid_request_type, prediction_before_signup, duplicate_natural_key`, `invalid_signup_timestamp`)
and `conflicting_duplicate_{event,customer,prediction}_id`. A row may carry several reasons separated by `;`.
Exact replays are `category = duplicate` with reason `duplicate_replay`.
