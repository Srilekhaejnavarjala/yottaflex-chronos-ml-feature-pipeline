# Design trade-offs, limitations, and interview answers

## Why Pandas + SQLite
Local, reproducible, zero infrastructure. SQLite gives real transactions, primary keys and
SQL-checkable invariants. Processing/persistence are separate modules, so either can be replaced.
For millions of events I would move event history to a columnar store (Parquet/DuckDB/warehouse),
compute windows with an as-of/range join, and publish via partitioned, idempotent merges.

## Known limitations (be upfront about these)
* **Batch-snapshot contract.** Each run's input is the authoritative full snapshot. No incremental load,
  no deletes. If a stored event's content changes between runs it is reported as `updated` (and logged as a
  warning); no correction history is kept. Use a separate DB per dataset (re-using one DB across different
  datasets leaves stale rows because publishing is additive).
* **Canonical-event persistence is part of the atomic publish stage** (not a separate stage), so events are never
  stored without their outputs.
* **Reconciliation runs after publish.** If it fails the run is marked `failed` and no files are exported,
  but the committed DB rows remain until a corrected rerun.
* Run statuses are `success`/`failed` only (no `partial_success`): publishing is all-or-nothing.
* Labels use `event_timestamp` only; they rely on the `data_complete_through` watermark for completeness.
  If no watermark is known, every label is `incomplete` (conservative).
* Feature/label column names embed `30d`/`7d`; changing windows needs a new feature_version and column set
  (the code refuses to run otherwise).
* Single currency (USD), static customer attributes, amounts as floats rounded to 2 dp.
* `source_row_number` is a file line number; quoted multi-line CSV fields would shift it.
* Only tested on Python 3.12 / pandas 3.0.6 / 1 CPU core. Memory was not measured.
* Publishing uses Python row loops for diff/upsert (2.6 s of the 9.7 s total at 100k events).

## Questions you should be ready for
**What is target leakage?** Features containing information not available when the prediction is made.
**Why is event time alone insufficient?** An event can occur before `T` but arrive after `T`; the model could
not have known it. Hence the second clock `available_at <= T`.
**Why is the lower bound exclusive?** So consecutive 30-day windows tile without double-counting an event at exactly
`T - 30d`, and "30 days" means at most 30 days of data. The rule lives in one place and is boundary-tested.
**Late-arriving events?** Excluded from any prediction made before arrival, included for predictions at/after it.
**Why labels separate?** `build_features` rejects label columns outright; features and labels only meet in
`assemble_training_dataset`, and inference outputs carry no label columns.
**Incomplete horizon?** `label_status=incomplete`, target NULL, never 0.
**Duplicates?** Same ID + same content: replay, first kept. Same ID + different content: all copies quarantined.
Different IDs: never deduplicated.
**Idempotency?** Upserts on `(prediction_id, feature_version)` etc.; rerun reports 0 inserted / 0 updated.
**Publish fails halfway?** One transaction; rollback; run marked failed at stage `publish`.
**Backfill?** Provide the prediction requests (with historical timestamps) and the event snapshot; the same
point-in-time rules apply. A changed feature definition gets a new `feature_version` (separate rows by key).
**What was expensive?** Filtering the whole transaction table per request (O(requests x events)).
**Optimization?** Sort once, locate each window with binary search per customer, filter only the window.
Measured 16x-129x faster with identical output (see README).
