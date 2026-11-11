"""Source table contracts: required columns, controlled vocabularies, grain."""

CUSTOMER_COLUMNS = ["customer_id", "signup_timestamp", "region", "customer_segment"]
CUSTOMER_REQUIRED = ["customer_id", "signup_timestamp"]

TRANSACTION_COLUMNS = [
    "event_id", "customer_id", "event_timestamp", "available_at",
    "event_type", "event_status", "amount", "currency",
]
TRANSACTION_REQUIRED = ["event_id", "customer_id", "event_timestamp", "available_at",
                        "event_type", "event_status"]

PREDICTION_COLUMNS = ["prediction_id", "customer_id", "prediction_timestamp", "request_type"]
PREDICTION_REQUIRED = PREDICTION_COLUMNS

EVENT_TYPES = {"purchase", "refund", "login", "cart_add", "cancellation"}
EVENT_STATUSES = {"completed", "pending", "failed", "cancelled"}
REQUEST_TYPES = {"training", "inference"}

# Event types that must carry amount + currency.
MONETARY_EVENT_TYPES = {"purchase", "refund"}

# Feature output column order is part of the contract (parity tests rely on it).
FEATURE_COLUMNS = [
    "prediction_id", "customer_id", "prediction_timestamp", "feature_version",
    "purchase_count_30d", "purchase_amount_sum_30d", "purchase_amount_avg_30d",
    "days_since_last_purchase", "refund_count_30d", "login_count_30d",
    "customer_tenure_days",
]
LABEL_COLUMNS = [
    "prediction_id", "customer_id", "prediction_timestamp", "label_window_end",
    "target_purchase_next_7d", "label_status", "label_version",
]

LABEL_LEAK_COLUMNS = {"target_purchase_next_7d", "label_status", "label_window_end"}
LABEL_STATUSES = {"complete", "incomplete"}
