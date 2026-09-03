"""
Data Quality Auditor — Colorful Web App
-------------------------------------------
Charts, 3D outlier explorer, colorful gradient cards, prioritized fix
list, and one-click auto-clean.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from data_quality_auditor import DataQualityAuditor

st.set_page_config(page_title="Data Quality Auditor", page_icon="🔍", layout="wide")

# ---------------- Global styling ----------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 50%, #00c6ff 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 18px rgba(37,117,252,0.25);
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }

    .card-blue, .card-purple, .card-orange, .card-green, .card-pink {
        padding: 1rem 1.3rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 0.8rem;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .card-blue   { background: linear-gradient(120deg, #2193b0, #6dd5ed); }
    .card-purple { background: linear-gradient(120deg, #834d9b, #d04ed6); }
    .card-orange { background: linear-gradient(120deg, #f7971e, #ffd200); color:#2b2b2b; }
    .card-green  { background: linear-gradient(120deg, #11998e, #38ef7d); }
    .card-pink   { background: linear-gradient(120deg, #ff5f6d, #ffc371); }

    div[data-testid="stMetric"] {
        background: #f8f9fb;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 0.8rem;
    }
    button[data-baseweb="tab"] { font-weight: 600; }
</style>
<div class="main-header">
    <h1>🔍 Data Quality Score Auditor</h1>
    <p>Upload any CSV — get a health score, colorful interactive charts, a 3D outlier explorer, and a prioritized fix list.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


def score_to_grade(health):
    if health >= 90: return "A — Excellent", "green"
    if health >= 75: return "B — Good", "blue"
    if health >= 60: return "C — Needs Attention", "orange"
    if health >= 40: return "D — Poor", "orange"
    return "F — Critical", "red"


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read this file as a CSV: {e}")
        st.stop()

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

    auditor = DataQualityAuditor(df)
    auditor.run_all_checks()
    health = auditor.results["health_score"]
    grade, grade_color = score_to_grade(health)

    tab_overview, tab_charts, tab_3d, tab_fixes, tab_clean = st.tabs(
        ["📊 Overview", "📈 Charts", "🧊 3D Outliers", "📋 Fix List", "🛠️ Auto-Clean"]
    )

    # ================= OVERVIEW =================
    with tab_overview:
        st.markdown(f'<div class="card-blue">Health Score: {health}/100 — {grade}</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Health Score", f"{health}/100")
            st.progress(min(int(health), 100))
        with col2:
            breakdown = pd.DataFrame([
                {"Dimension": "Missing Values", "Score": auditor.results["missing"]["score"], "Weight": "30%"},
                {"Dimension": "Duplicates", "Score": auditor.results["duplicates"]["score"], "Weight": "20%"},
                {"Dimension": "Outliers", "Score": auditor.results["outliers"]["score"], "Weight": "20%"},
                {"Dimension": "Consistency", "Score": auditor.results["consistency"]["score"], "Weight": "30%"},
            ])
            st.dataframe(breakdown, use_container_width=True, hide_index=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Missing Cells", auditor.results["missing"]["total_missing"])
        m2.metric("Duplicate Rows", auditor.results["duplicates"]["duplicate_rows"])
        m3.metric("Outliers", auditor.results["outliers"]["total_outliers"])
        m4.metric("Format Issues", auditor.results["consistency"]["total_format_issues"])

        with st.expander("Preview the raw data"):
            st.dataframe(df.head(20), use_container_width=True)

        report_md = auditor.generate_report(dataset_name=uploaded_file.name)
        st.download_button("⬇ Download Full Report (Markdown)", data=report_md,
                            file_name="data_quality_report.md", mime="text/markdown")

    # ================= CHARTS =================
    with tab_charts:
        st.markdown('<div class="card-purple">Interactive Charts</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Missing Values by Column")
            missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            missing_pct = missing_pct[missing_pct > 0]
            if len(missing_pct) > 0:
                fig = px.bar(x=missing_pct.values, y=missing_pct.index, orientation="h",
                              labels={"x": "% Missing", "y": "Column"},
                              color=missing_pct.values, color_continuous_scale="Sunsetdark")
                fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No missing values found.")

        with c2:
            st.subheader("Issues by Severity")
            issues = auditor.prioritized_fix_list()
            if issues:
                sev_counts = pd.Series([i["severity"] for i in issues]).value_counts()
                fig = px.pie(values=sev_counts.values, names=sev_counts.index, color=sev_counts.index,
                              color_discrete_map={"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"},
                              hole=0.45)
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No issues found.")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Outlier Count by Column")
            by_col = auditor.results["outliers"]["by_column"]
            if by_col:
                fig = px.bar(x=list(by_col.keys()), y=list(by_col.values()),
                              labels={"x": "Column", "y": "Outlier Count"},
                              color=list(by_col.values()), color_continuous_scale="Plasma")
                fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No outliers found in numeric columns.")

        with c4:
            st.subheader("Numeric Correlation Heatmap")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Turbo", zmin=-1, zmax=1, aspect="auto")
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for a correlation heatmap.")

    # ================= 3D OUTLIERS =================
    with tab_3d:
        st.markdown('<div class="card-green">3D Outlier Explorer</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            st.info("This dataset needs at least 3 numeric columns to plot in 3D. "
                    f"Found: {len(numeric_cols)} ({', '.join(numeric_cols) if numeric_cols else 'none'}).")
        else:
            colx, coly, colz = st.columns(3)
            x_axis = colx.selectbox("X axis", numeric_cols, index=0, key="outlier_x")
            y_axis = coly.selectbox("Y axis", numeric_cols, index=1, key="outlier_y")
            z_axis = colz.selectbox("Z axis", numeric_cols, index=2, key="outlier_z")

            plot_df = df[[x_axis, y_axis, z_axis]].dropna().copy()

            def outlier_mask(series):
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    return pd.Series(False, index=series.index)
                return (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)

            is_outlier = outlier_mask(plot_df[x_axis]) | outlier_mask(plot_df[y_axis]) | outlier_mask(plot_df[z_axis])
            plot_df["status"] = np.where(is_outlier, "Outlier", "Normal")

            fig = px.scatter_3d(plot_df, x=x_axis, y=y_axis, z=z_axis, color="status",
                                 color_discrete_map={"Normal": "#2575fc", "Outlier": "#e74c3c"}, opacity=0.75)
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(height=650, legend_title_text="Point Status")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{int(is_outlier.sum())} of {len(plot_df)} rows flagged as an outlier on at least one axis. Drag to rotate.")

    # ================= FIX LIST =================
    with tab_fixes:
        st.markdown('<div class="card-purple">Prioritized Fix List</div>', unsafe_allow_html=True)
        issues = auditor.prioritized_fix_list()

        if not issues:
            st.info("No issues found — this dataset is clean!")
        else:
            issues_df = pd.DataFrame(issues)[["severity", "check", "column", "affected_rows", "pct_affected", "fix"]]
            issues_df.columns = ["Priority", "Check", "Column", "Rows Affected", "% Affected", "Recommended Fix"]

            def highlight_severity(row):
                color_map = {"HIGH": "#ffcccc", "MEDIUM": "#fff3cd", "LOW": "#d4edda"}
                return [f"background-color: {color_map.get(row['Priority'], '')}"] * len(row)

            st.dataframe(issues_df.style.apply(highlight_severity, axis=1),
                         use_container_width=True, hide_index=True)

    # ================= AUTO-CLEAN =================
    with tab_clean:
        st.markdown('<div class="card-blue">Auto-Clean This Dataset</div>', unsafe_allow_html=True)
        st.caption(
            "Applies safe, explainable fixes only: fills missing values with median/mode, "
            "removes exact duplicates, trims whitespace, standardizes casing on categorical columns, "
            "and caps (not deletes) statistical outliers. No rows are ever deleted."
        )

        if st.button("Run Auto-Clean", type="primary"):
            cleaned_df, change_log = auditor.auto_clean()
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
    st.info("👆 Upload a CSV file above to get started.")
