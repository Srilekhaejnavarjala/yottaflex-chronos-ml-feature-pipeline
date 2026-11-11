
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Project configuration
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline


st.set_page_config(
    page_title="Chronos | Feature Pipeline",
    page_icon="⏱️",
    layout="wide"
)

st.title("⏱️ Chronos")
st.caption("Point-in-Time Correct ML Feature Engineering Pipeline")

OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
FEATURE_DIR = OUTPUT_DIR / "features"
LABEL_DIR = OUTPUT_DIR / "labels"
REJECT_DIR = OUTPUT_DIR / "rejects"


def get_latest_report():
    if not REPORT_DIR.exists():
        return None

    reports = list(REPORT_DIR.glob("*.json"))

    if not reports:
        return None

    latest = max(reports, key=lambda file: file.stat().st_mtime)

    with open(latest, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)

    return None


def display_overview(summary):
    sources = summary.get("sources", {})
    transactions = sources.get("transactions", {})
    predictions = sources.get("prediction_requests", {})

    st.subheader("Pipeline Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Input Events", transactions.get("input_rows", 0))
    c2.metric("Valid Events", transactions.get("valid_rows", 0))
    c3.metric("Feature Rows", summary.get("feature_rows", 0))
    c4.metric("Prediction Requests", predictions.get("input_rows", 0))

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Complete Labels", summary.get("complete_label_rows", 0))
    c6.metric("Incomplete Labels", summary.get("incomplete_label_rows", 0))
    c7.metric("Rejected Events", transactions.get("rejected_rows", 0))
    c8.metric("Duration", f"{summary.get('duration_seconds', 0):.3f} sec")

    st.divider()

    st.subheader("Latest Execution")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Run ID**")
        st.code(summary.get("run_id", "N/A"))

        st.write("**Feature Version**")
        st.write(summary.get("feature_version", "N/A"))

    with col2:
        status = summary.get("status", "unknown").upper()

        if status == "SUCCESS":
            st.success(f"Pipeline Status: {status}")
        else:
            st.error(f"Pipeline Status: {status}")

        st.write("**Data Complete Through**")
        st.write(summary.get("data_complete_through") or "N/A")

    stage_seconds = summary.get("stage_seconds", {})

    if stage_seconds:
        st.subheader("Stage Execution Time")
        stage_df = pd.DataFrame(
            list(stage_seconds.items()),
            columns=["Stage", "Duration (seconds)"]
        ).set_index("Stage")

        st.bar_chart(stage_df)


def display_features(summary):
    version = summary.get("feature_version", "v1")

    feature_file = FEATURE_DIR / f"features_{version}.csv"
    inference_file = FEATURE_DIR / f"inference_features_{version}.csv"
    training_file = FEATURE_DIR / f"training_dataset_{version}.csv"

    choice = st.selectbox(
        "Select dataset",
        ["All Features", "Inference Features", "Training Dataset"]
    )

    paths = {
        "All Features": feature_file,
        "Inference Features": inference_file,
        "Training Dataset": training_file
    }

    df = load_csv(paths[choice])

    if df is not None:
        st.write(f"Total records: **{len(df)}**")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Dataset not found. Run the pipeline first.")


def display_labels(summary):
    version = summary.get("label_version", "v1")
    path = LABEL_DIR / f"labels_{version}.csv"

    df = load_csv(path)

    if df is not None:
        st.write(f"Total label records: **{len(df)}**")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Label dataset not found. Run the pipeline first.")


def display_quality(summary):
    st.subheader("Data Quality & Reconciliation")

    sources = summary.get("sources", {})

    for name, counts in sources.items():
        with st.expander(name.replace("_", " ").title()):
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Input", counts.get("input_rows", 0))
            c2.metric("Valid", counts.get("valid_rows", 0))
            c3.metric("Rejected", counts.get("rejected_rows", 0))
            c4.metric("Duplicates", counts.get("duplicate_rows", 0))

    st.divider()

    checks = summary.get("checks", [])

    passed = sum(check.get("passed", False) for check in checks)
    failed = len(checks) - passed

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Checks", len(checks))
    c2.metric("Passed", passed)
    c3.metric("Failed", failed)

    for check in checks:
        if check.get("passed"):
            st.success(f"{check['name']} — {check.get('detail', '')}")
        else:
            st.error(f"{check['name']} — {check.get('detail', '')}")


def display_rejects(summary):
    run_id = summary.get("run_id", "")
    path = REJECT_DIR / f"rejects_{run_id}.csv"

    df = load_csv(path)

    if df is not None:
        if df.empty:
            st.success("No rejected records in this run.")
        else:
            st.warning(f"{len(df)} rejected records")
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No rejection file available for this run.")


# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Run Pipeline",
        "Feature Explorer",
        "Labels",
        "Data Quality",
        "Rejected Records"
    ]
)

summary = st.session_state.get("summary")

if summary is None:
    summary = get_latest_report()


if page == "Dashboard":
    if summary:
        display_overview(summary)
    else:
        st.info("No pipeline execution found. Run the pipeline to get started.")


elif page == "Run Pipeline":
    st.subheader("Execute Chronos Pipeline")

    st.write(
        "Execute the existing ingestion, validation, feature engineering, "
        "label generation, publishing, reconciliation and export stages."
    )

    if st.button("▶ Run Pipeline", type="primary"):
        try:
            with st.spinner("Executing Chronos pipeline..."):
                summary = run_pipeline(
                    customers_path="data/sample/customers.csv",
                    transactions_paths=["data/sample/transactions.csv"],
                    predictions_path="data/sample/prediction_requests.csv",
                    db_path="outputs/chronos.db",
                    out_dir="outputs"
                )

                st.session_state["summary"] = summary

            st.success("Pipeline executed successfully!")

        except Exception as exc:
            st.session_state["summary"] = get_latest_report()
            st.error(f"Pipeline execution failed: {exc}")

    if st.session_state.get("summary"):
        display_overview(st.session_state["summary"])


elif page == "Feature Explorer":
    st.subheader("Feature Explorer")

    if summary:
        display_features(summary)
    else:
        st.info("Run the pipeline first.")


elif page == "Labels":
    st.subheader("Generated Target Labels")

    if summary:
        display_labels(summary)
    else:
        st.info("Run the pipeline first.")


elif page == "Data Quality":
    if summary:
        display_quality(summary)
    else:
        st.info("No execution report available.")


elif page == "Rejected Records":
    st.subheader("Rejected Records")

    if summary:
        display_rejects(summary)
    else:
        st.info("No execution report available.")
