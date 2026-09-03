"""
Data Quality Score Auditor
---------------------------
Scans a CSV (or a pandas DataFrame) and produces:
  - Missing value analysis
  - Duplicate row detection
  - Outlier detection (IQR method, numeric columns)
  - Format consistency checks (whitespace, casing, date formats)
  - A single 0-100 "Health Score"
  - A prioritized, ranked fix list

Usage:
    python data_quality_auditor.py path/to/data.csv
    python data_quality_auditor.py path/to/data.csv --output-dir ./reports
"""

import argparse
import json
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

DATE_CANDIDATE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}$",        # 2024-01-31
    r"^\d{2}/\d{2}/\d{4}$",        # 01/31/2024
    r"^\d{2}-\d{2}-\d{4}$",        # 31-01-2024
    r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # 1/31/24
]


class DataQualityAuditor:
    def __init__(self, df: pd.DataFrame, weights: dict | None = None):
        self.df = df
        self.n_rows, self.n_cols = df.shape
        self.total_cells = max(self.n_rows * self.n_cols, 1)
        # How much each dimension counts toward the final health score.
        # Tune these if your domain cares more about one issue than another.
        self.weights = weights or {
            "missing": 0.30,
            "duplicates": 0.20,
            "outliers": 0.20,
            "consistency": 0.30,
        }
        self.issues = []      # flat list of every individual issue found
        self.results = {}     # per-dimension summary stats + sub-score

    # ---------- severity helper ----------
    @staticmethod
    def _severity(pct_affected: float) -> str:
        if pct_affected >= 15:
            return "HIGH"
        if pct_affected >= 5:
            return "MEDIUM"
        return "LOW"

    # ---------- 1. Missing values ----------
    def check_missing_values(self):
        missing = self.df.isnull().sum()
        total_missing = int(missing.sum())
        pct_overall = round((total_missing / self.total_cells) * 100, 2)

        for col, count in missing.items():
            if count == 0:
                continue
            col_pct = round((count / self.n_rows) * 100, 2)
            self.issues.append({
                "check": "Missing Values",
                "column": col,
                "affected_rows": int(count),
                "pct_affected": col_pct,
                "severity": self._severity(col_pct),
                "fix": f"Impute or drop '{col}' — {count} missing ({col_pct}%)",
            })

        score = max(0, 100 - pct_overall * 2)  # missing data is penalized heavily
        self.results["missing"] = {
            "total_missing": total_missing,
            "pct_missing_overall": pct_overall,
            "score": round(score, 2),
        }
        return self.results["missing"]

    # ---------- 2. Duplicates ----------
    def check_duplicates(self):
        dup_count = int(self.df.duplicated().sum())
        pct = round((dup_count / self.n_rows) * 100, 2) if self.n_rows else 0

        if dup_count > 0:
            self.issues.append({
                "check": "Duplicate Rows",
                "column": "ALL COLUMNS",
                "affected_rows": dup_count,
                "pct_affected": pct,
                "severity": self._severity(pct),
                "fix": f"Remove {dup_count} exact duplicate row(s) ({pct}% of dataset)",
            })

        score = max(0, 100 - pct * 3)  # duplicates are cheap to fix, but signal bigger pipeline issues
        self.results["duplicates"] = {
            "duplicate_rows": dup_count,
            "pct_duplicates": pct,
            "score": round(score, 2),
        }
        return self.results["duplicates"]

    # ---------- 3. Outliers (IQR method) ----------
    def check_outliers(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        total_numeric_cells = max(self.n_rows * len(numeric_cols), 1)
        by_column = {}

        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) < 4:
                continue
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            if len(outliers) == 0:
                continue

            pct = round((len(outliers) / self.n_rows) * 100, 2)
            total_outliers += len(outliers)
            by_column[col] = len(outliers)
            self.issues.append({
                "check": "Outliers",
                "column": col,
                "affected_rows": int(len(outliers)),
                "pct_affected": pct,
                "severity": self._severity(pct),
                "fix": f"Review {len(outliers)} outlier(s) in '{col}' outside [{lower:.2f}, {upper:.2f}]",
            })

        pct_overall = round((total_outliers / total_numeric_cells) * 100, 2)
        score = max(0, 100 - pct_overall * 2)
        self.results["outliers"] = {
            "total_outliers": total_outliers,
            "pct_outliers_overall": pct_overall,
            "score": round(score, 2),
            "by_column": by_column,
        }
        return self.results["outliers"]

    # ---------- 4. Format consistency ----------
    def _looks_like_dates(self, series: pd.Series) -> bool:
        sample = series.head(20)
        hits = sum(any(re.match(p, str(v).strip()) for p in DATE_CANDIDATE_PATTERNS) for v in sample)
        return hits >= max(3, len(sample) // 2)

    def _count_mixed_date_formats(self, series: pd.Series) -> int:
        def fmt_signature(v):
            v = str(v).strip()
            for p in DATE_CANDIDATE_PATTERNS:
                if re.match(p, v):
                    return p
            return None

        sigs = series.map(fmt_signature)
        sigs = sigs.dropna()
        if sigs.nunique() <= 1:
            return 0
        majority = sigs.value_counts().idxmax()
        return int((sigs != majority).sum())

    def check_format_consistency(self):
        string_cols = self.df.select_dtypes(include=["object"]).columns
        total_issues = 0
        total_string_cells = max(self.n_rows * len(string_cols), 1)

        for col in string_cols:
            series = self.df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            # a) leading/trailing whitespace
            ws_mask = series != series.str.strip()
            ws_count = int(ws_mask.sum())
            if ws_count > 0:
                pct = round(ws_count / self.n_rows * 100, 2)
                total_issues += ws_count
                self.issues.append({
                    "check": "Format Consistency",
                    "column": col,
                    "affected_rows": ws_count,
                    "pct_affected": pct,
                    "severity": self._severity(pct),
                    "fix": f"Trim stray whitespace in '{col}' ({ws_count} rows)",
                })

            # b) inconsistent casing (e.g. "USA" vs "usa" vs "Usa")
            unique_raw = series.nunique()
            unique_lower = series.str.lower().nunique()
            if unique_lower < unique_raw:
                collapse = unique_raw - unique_lower
                pct = round(collapse / self.n_rows * 100, 2)
                total_issues += collapse
                self.issues.append({
                    "check": "Format Consistency",
                    "column": col,
                    "affected_rows": collapse,
                    "pct_affected": pct,
                    "severity": self._severity(pct),
                    "fix": f"Standardize casing in '{col}' — {unique_raw} unique values collapse to {unique_lower}",
                })

            # c) mixed date formats
            if self._looks_like_dates(series):
                mixed = self._count_mixed_date_formats(series)
                if mixed > 0:
                    pct = round(mixed / self.n_rows * 100, 2)
                    total_issues += mixed
                    self.issues.append({
                        "check": "Format Consistency",
                        "column": col,
                        "affected_rows": mixed,
                        "pct_affected": pct,
                        "severity": self._severity(pct),
                        "fix": f"Standardize date format in '{col}' to one convention (e.g. ISO 8601)",
                    })

        pct_overall = round((total_issues / total_string_cells) * 100, 2)
        score = max(0, 100 - pct_overall * 1.5)
        self.results["consistency"] = {
            "total_format_issues": total_issues,
            "pct_issues_overall": pct_overall,
            "score": round(score, 2),
        }
        return self.results["consistency"]

    # ---------- run everything ----------
    def run_all_checks(self):
        self.check_missing_values()
        self.check_duplicates()
        self.check_outliers()
        self.check_format_consistency()
        return self.compute_health_score()

    def compute_health_score(self):
        health = sum(self.results[k]["score"] * w for k, w in self.weights.items())
        self.results["health_score"] = round(health, 1)
        return self.results["health_score"]

    def prioritized_fix_list(self):
        sev_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        return sorted(
            self.issues,
            key=lambda i: (sev_rank[i["severity"]], -i["affected_rows"]),
        )

    # ---------- auto-clean ----------
    def auto_clean(self, cap_outliers=True, standardize_low_cardinality_text=True):
        """
        Returns (cleaned_df, change_log).
        Applies safe, explainable fixes only:
          - drops exact duplicate rows
          - fills missing numeric with column median
          - fills missing text with column mode (or 'Unknown' if no mode)
          - trims whitespace on all text columns
          - standardizes casing ONLY on low-cardinality text columns
            (heuristic: fewer than 50 unique values AND unique values < 20% of rows)
            to avoid mangling free-text fields like names or descriptions
          - caps (winsorizes) numeric outliers at the IQR boundary instead of deleting rows
        Never deletes rows for missing data, and never deletes outlier rows.
        """
        df = self.df.copy()
        log = []

        # 1. Duplicates - always safe to drop
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed > 0:
            log.append(f"Removed {removed} exact duplicate row(s).")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=["object"]).columns

        # 2. Missing numeric -> median
        for col in numeric_cols:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.append(f"Filled {n_missing} missing value(s) in '{col}' with median ({median_val:.2f}).")

        # 3. Missing text -> mode (or 'Unknown')
        for col in text_cols:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                mode_series = df[col].mode(dropna=True)
                fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                df[col] = df[col].fillna(fill_val)
                log.append(f"Filled {n_missing} missing value(s) in '{col}' with '{fill_val}'.")

        # 4. Trim whitespace on all text columns
        for col in text_cols:
            trimmed = df[col].astype(str).str.strip()
            changed = (trimmed != df[col].astype(str)).sum()
            if changed > 0:
                df[col] = trimmed
                log.append(f"Trimmed whitespace on {changed} value(s) in '{col}'.")

        # 5. Standardize casing on low-cardinality text columns only
        if standardize_low_cardinality_text:
            for col in text_cols:
                n_unique = df[col].nunique()
                if n_unique == 0:
                    continue
                is_low_cardinality = n_unique < 50 and (n_unique / max(len(df), 1)) < 0.2
                if not is_low_cardinality:
                    continue
                lowered = df[col].astype(str).str.lower()
                if lowered.nunique() < n_unique:
                    mapping = (
                        df[col].astype(str).groupby(lowered)
                        .agg(lambda x: x.value_counts().idxmax())
                    )
                    new_col = lowered.map(mapping)
                    changed = (new_col != df[col].astype(str)).sum()
                    if changed > 0:
                        df[col] = new_col
                        log.append(f"Standardized casing on {changed} value(s) in '{col}' ({n_unique} unique -> {lowered.nunique()} unique).")

        # 6. Cap outliers (winsorize) instead of deleting
        if cap_outliers:
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) < 4:
                    continue
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_capped > 0:
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    log.append(f"Capped {n_capped} outlier(s) in '{col}' to range [{lower:.2f}, {upper:.2f}] instead of deleting them.")

        if not log:
            log.append("No safe automatic fixes were needed - dataset was already clean.")

        return df, log

    # ---------- reporting ----------
    def generate_report(self, dataset_name="dataset") -> str:
        health = self.results.get("health_score", self.compute_health_score())
        grade = (
            "A - Excellent" if health >= 90 else
            "B - Good" if health >= 75 else
            "C - Needs Attention" if health >= 60 else
            "D - Poor" if health >= 40 else
            "F - Critical"
        )

        lines = []
        lines.append(f"# Data Quality Report — {dataset_name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Rows: {self.n_rows:,} | Columns: {self.n_cols}")
        lines.append("")
        lines.append(f"## Health Score: {health}/100  ({grade})")
        lines.append("")
        lines.append("| Dimension | Score | Weight |")
        lines.append("|---|---|---|")
        for k in ["missing", "duplicates", "outliers", "consistency"]:
            r = self.results[k]
            lines.append(f"| {k.capitalize()} | {r['score']}/100 | {int(self.weights[k]*100)}% |")
        lines.append("")

        lines.append("## Summary")
        m, d, o, c = (self.results[k] for k in ["missing", "duplicates", "outliers", "consistency"])
        lines.append(f"- **Missing values:** {m['total_missing']} cells ({m['pct_missing_overall']}% of all data)")
        lines.append(f"- **Duplicate rows:** {d['duplicate_rows']} ({d['pct_duplicates']}%)")
        lines.append(f"- **Outliers:** {o['total_outliers']} values across {len(o['by_column'])} numeric column(s)")
        lines.append(f"- **Format issues:** {c['total_format_issues']} inconsistencies across text columns")
        lines.append("")

        lines.append("## Prioritized Fix List")
        lines.append("| Priority | Check | Column | Rows Affected | % | Recommended Fix |")
        lines.append("|---|---|---|---|---|---|")
        for i, issue in enumerate(self.prioritized_fix_list(), 1):
            lines.append(
                f"| {i} ({issue['severity']}) | {issue['check']} | {issue['column']} | "
                f"{issue['affected_rows']} | {issue['pct_affected']}% | {issue['fix']} |"
            )

        if not self.issues:
            lines.append("| - | - | - | - | - | No issues found. |")

        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "dataset_shape": {"rows": self.n_rows, "columns": self.n_cols},
            "health_score": self.results.get("health_score", self.compute_health_score()),
            "dimension_scores": {k: self.results[k] for k in ["missing", "duplicates", "outliers", "consistency"]},
            "issues": self.prioritized_fix_list(),
        }
        return json.dumps(payload, indent=2, default=str)


def audit_csv(path: str, output_dir: str = "."):
    df = pd.read_csv(path)
    auditor = DataQualityAuditor(df)
    auditor.run_all_checks()

    name = path.split("/")[-1]
    report_md = auditor.generate_report(dataset_name=name)
    report_json = auditor.to_json()

    md_path = f"{output_dir}/data_quality_report.md"
    json_path = f"{output_dir}/data_quality_report.json"
    with open(md_path, "w") as f:
        f.write(report_md)
    with open(json_path, "w") as f:
        f.write(report_json)

    return auditor, md_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Quality Score Auditor")
    parser.add_argument("csv_path", help="Path to the CSV file to audit")
    parser.add_argument("--output-dir", default=".", help="Where to save the report")
    args = parser.parse_args()

    auditor, md_path, json_path = audit_csv(args.csv_path, args.output_dir)
    print(f"Health Score: {auditor.results['health_score']}/100")
    print(f"Report saved to: {md_path}")
    print(f"JSON saved to: {json_path}")
