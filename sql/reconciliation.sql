-- Each named query returns ONE number: the count of violations. Expected value: 0.
-- Parameters: :feature_version  :label_version  :watermark (ISO-8601 or NULL)  :run_id
-- Run manually:  sqlite3 outputs/chronos.db  then paste a query with parameters bound,
-- or let `python -m src.pipeline` run them all automatically.

-- name: duplicate_feature_keys
SELECT COUNT(*) FROM (SELECT 1 FROM customer_features GROUP BY prediction_id, feature_version HAVING COUNT(*) > 1);

-- name: duplicate_label_keys
SELECT COUNT(*) FROM (SELECT 1 FROM training_labels GROUP BY prediction_id, label_version HAVING COUNT(*) > 1);

-- name: features_without_prediction_request
SELECT COUNT(*) FROM customer_features f LEFT JOIN prediction_requests p ON p.prediction_id = f.prediction_id
WHERE p.prediction_id IS NULL;

-- name: features_unknown_customer
SELECT COUNT(*) FROM customer_features f LEFT JOIN customers c ON c.customer_id = f.customer_id
WHERE c.customer_id IS NULL;

-- name: features_customer_mismatch
SELECT COUNT(*) FROM customer_features f JOIN prediction_requests p ON p.prediction_id = f.prediction_id
WHERE p.customer_id <> f.customer_id OR p.prediction_timestamp <> f.prediction_timestamp;

-- name: requests_missing_features
SELECT COUNT(*) FROM prediction_requests p
LEFT JOIN customer_features f ON f.prediction_id = p.prediction_id AND f.feature_version = :feature_version
WHERE f.prediction_id IS NULL;

-- name: training_requests_missing_labels
SELECT COUNT(*) FROM prediction_requests p
LEFT JOIN training_labels l ON l.prediction_id = p.prediction_id AND l.label_version = :label_version
WHERE p.request_type = 'training' AND l.prediction_id IS NULL;

-- name: labels_for_non_training_requests
SELECT COUNT(*) FROM training_labels l JOIN prediction_requests p ON p.prediction_id = l.prediction_id
WHERE p.request_type <> 'training';

-- name: complete_labels_beyond_watermark
SELECT COUNT(*) FROM training_labels
WHERE label_status = 'complete' AND (:watermark IS NULL OR label_window_end > :watermark);

-- name: incomplete_labels_with_covered_horizon
SELECT COUNT(*) FROM training_labels
WHERE label_status = 'incomplete' AND :watermark IS NOT NULL AND label_window_end <= :watermark;

-- name: rejects_missing_traceability
SELECT COUNT(*) FROM rejected_records
WHERE run_id = :run_id AND (source_file IS NULL OR source_file = '' OR source_row_number IS NULL
                            OR reject_reason IS NULL OR reject_reason = '');

-- name: events_available_before_occurrence
SELECT COUNT(*) FROM canonical_events WHERE available_at < event_timestamp;

-- name: events_unknown_customer
SELECT COUNT(*) FROM canonical_events e LEFT JOIN customers c ON c.customer_id = e.customer_id
WHERE c.customer_id IS NULL;
