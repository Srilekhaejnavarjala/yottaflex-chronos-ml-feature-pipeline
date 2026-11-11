import filecmp

from src.generate_data import generate


def test_same_seed_is_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, customers=20, events=200, seed=3, include_invalid=True)
    generate(b, customers=20, events=200, seed=3, include_invalid=True)
    for f in ("customers.csv", "transactions.csv", "prediction_requests.csv", "invalid_transactions.csv", "source_metadata.json"):
        assert filecmp.cmp(a / f, b / f, shallow=False), f


def test_different_seed_differs(tmp_path):
    generate(tmp_path / "a", customers=20, events=200, seed=1)
    generate(tmp_path / "b", customers=20, events=200, seed=2)
    assert not filecmp.cmp(tmp_path / "a" / "transactions.csv", tmp_path / "b" / "transactions.csv", shallow=False)


def test_coverage_of_required_cases(small_data):
    import pandas as pd
    tx = pd.read_csv(small_data / "transactions.csv", dtype=str, keep_default_na=False)
    assert {"purchase", "login", "refund", "cart_add"} <= set(tx["event_type"])
    assert {"pending", "failed", "cancelled", "completed"} <= set(tx["event_status"])
    assert tx["event_id"].str.contains("_lower_excl").any() and tx["event_id"].str.contains("_late_arrival").any()
    assert (tx["available_at"] > tx["event_timestamp"]).any()          # late arrivals present
    assert not tx["event_id"].is_unique is False                        # clean file has unique ids
    assert not tx["event_timestamp"].is_monotonic_increasing           # out-of-order rows
    preds = pd.read_csv(small_data / "prediction_requests.csv", dtype=str)
    assert {"training", "inference"} == set(preds["request_type"])
    cust = pd.read_csv(small_data / "customers.csv", dtype=str)
    assert set(cust["customer_id"]) - set(tx["customer_id"])           # customers with no events
    assert (small_data / "invalid_transactions.csv").exists()
