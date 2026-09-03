"""
Data Quality Auditor — Web App
--------------------------------
A drag-and-drop web interface for the Data Quality Auditor tool.
Run with:  streamlit run app.py
Then open the local URL it prints (usually http://localhost:8501)
"""

import streamlit as st
import pandas as pd

from data_quality_auditor import DataQualityAuditor

st.set_page_config(page_title="Data Quality Auditor", page_icon="🔍", layout="wide")

st.title("🔍 Data Quality Score Auditor")
st.write("Upload any CSV file to get an instant data quality report — missing values, duplicates, outliers, and formatting issues, ranked by priority.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read this file as a CSV: {e}")
        st.stop()

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    with st.expander("Preview the raw data"):
        st.dataframe(df.head(20), use_container_width=True)

    auditor = DataQualityAuditor(df)
    auditor.run_all_checks()
    health = auditor.results["health_score"]

    # ---- Health score header ----
    grade = (
        "A — Excellent" if health >= 90 else
        "B — Good" if health >= 75 else
        "C — Needs Attention" if health >= 60 else
        "D — Poor" if health >= 40 else
        "F — Critical"
    )
    color = "green" if health >= 75 else "orange" if health >= 50 else "red"

    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Health Score", f"{health}/100")
        st.markdown(f":{color}[**{grade}**]")
        st.progress(min(int(health), 100))

    with col2:
        st.subheader("Score Breakdown")
        breakdown = pd.DataFrame([
            {"Dimension": "Missing Values", "Score": auditor.results["missing"]["score"], "Weight": "30%"},
            {"Dimension": "Duplicates", "Score": auditor.results["duplicates"]["score"], "Weight": "20%"},
            {"Dimension": "Outliers", "Score": auditor.results["outliers"]["score"], "Weight": "20%"},
            {"Dimension": "Consistency", "Score": auditor.results["consistency"]["score"], "Weight": "30%"},
        ])
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    # ---- Summary metrics ----
    st.markdown("---")
    st.subheader("Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Missing Cells", auditor.results["missing"]["total_missing"])
    m2.metric("Duplicate Rows", auditor.results["duplicates"]["duplicate_rows"])
    m3.metric("Outliers", auditor.results["outliers"]["total_outliers"])
    m4.metric("Format Issues", auditor.results["consistency"]["total_format_issues"])

    # ---- Prioritized fix list ----
    st.markdown("---")
    st.subheader("📋 Prioritized Fix List")
    issues = auditor.prioritized_fix_list()

    if not issues:
        st.info("No issues found — this dataset is clean!")
    else:
        issues_df = pd.DataFrame(issues)[["severity", "check", "column", "affected_rows", "pct_affected", "fix"]]
        issues_df.columns = ["Priority", "Check", "Column", "Rows Affected", "% Affected", "Recommended Fix"]

        def highlight_severity(row):
            color_map = {"HIGH": "#ffcccc", "MEDIUM": "#fff3cd", "LOW": "#d4edda"}
            return [f"background-color: {color_map.get(row['Priority'], '')}"] * len(row)

        st.dataframe(
            issues_df.style.apply(highlight_severity, axis=1),
            use_container_width=True,
            hide_index=True,
        )

    # ---- Download report ----
    st.markdown("---")
    report_md = auditor.generate_report(dataset_name=uploaded_file.name)
    st.download_button(
        "⬇ Download Full Report (Markdown)",
        data=report_md,
        file_name="data_quality_report.md",
        mime="text/markdown",
    )

    # ---- Auto-clean ----
    st.markdown("---")
    st.subheader("🛠️ Auto-Clean This Dataset")
    st.caption(
        "Applies safe, explainable fixes only: fills missing values with median/mode, "
        "removes exact duplicates, trims whitespace, standardizes casing on categorical columns, "
        "and caps (not deletes) statistical outliers. No rows are ever deleted for missing data or outliers."
    )

    if st.button("Run Auto-Clean"):
        cleaned_df, change_log = auditor.auto_clean()

        # Re-audit the cleaned data to show the improvement
        cleaned_auditor = DataQualityAuditor(cleaned_df)
        cleaned_auditor.run_all_checks()
        new_health = cleaned_auditor.results["health_score"]

        c1, c2 = st.columns(2)
        c1.metric("Health Score Before", f"{health}/100")
        c2.metric("Health Score After", f"{new_health}/100", delta=round(new_health - health, 1))

        st.write("**What was changed:**")
        for line in change_log:
            st.write(f"- {line}")

        with st.expander("Preview cleaned data"):
            st.dataframe(cleaned_df.head(20), use_container_width=True)

        st.download_button(
            "⬇ Download Cleaned Dataset (CSV)",
            data=cleaned_df.to_csv(index=False),
            file_name=f"cleaned_{uploaded_file.name}",
            mime="text/csv",
        )

else:
    st.info("👆 Upload a CSV file above to get started. Try one of your test files: sample_messy_data.csv, titanic.csv, or movies.csv")
