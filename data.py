import pandas as pd
import numpy as np
import streamlit as st

BRANCHES = ["Orchard", "Tampines", "Jurong East", "Woodlands",
            "Bishan", "Bugis", "Toa Payoh", "Clementi"]

PLOT_CATALOGUE = {
    "line":    "Monthly trend over time for one metric",
    "bar":     "Branch comparison for one metric at the latest period",
    "area":    "Filled area trend over time — good for volume/demand metrics",
    "scatter": "Correlation between two metrics across branches (requires metric_y)",
    "heatmap": "Branch × month intensity grid — reveals patterns across time and branches",
    "ranking": "Horizontal bar ranking all branches for one metric",
}

METRIC_META = {
    "avg_wait_time":       {"label": "Avg Wait Time",       "unit": "min",       "lower_is_better": True},
    "avg_handling_time":   {"label": "Avg Handling Time",   "unit": "min",       "lower_is_better": True},
    "queue_tokens":        {"label": "Queue Volume",        "unit": "tokens",    "lower_is_better": False},
    "missed_queue":        {"label": "Missed Queue Count",  "unit": "customers", "lower_is_better": True},
    "total_transactions":  {"label": "Total Transactions",  "unit": "txns",      "lower_is_better": False},
    "senior_pct":          {"label": "Senior Customers",    "unit": "%",         "lower_is_better": False},
    "corporate_clients":   {"label": "Corporate Clients",   "unit": "clients",   "lower_is_better": False},
    "counter_utilization": {"label": "Counter Utilisation", "unit": "%",         "lower_is_better": False},
    "staff_seedling":      {"label": "Seedling Staff",      "unit": "headcount", "lower_is_better": False},
    "staff_sapling":       {"label": "Sapling Staff",       "unit": "headcount", "lower_is_better": False},
    "staff_mature":        {"label": "Mature Staff",        "unit": "headcount", "lower_is_better": False},
}

# Per-branch parameters:
# (wait, handle, tokens, missed, txn, senior_pct, corp, util, seedling, sapling, mature,
#  wait_trend, missed_trend, txn_trend, corp_trend, util_trend)
_PARAMS = {
    "Orchard":     (10, 7, 280, 12, 7200, 22, 180, 82, 3, 5, 4, -0.20, -0.25, +0.22, +0.15, +0.10),
    "Tampines":    (13, 8, 320, 20, 6500, 28,  90, 75, 4, 5, 3, -0.12, -0.18, +0.20, +0.08, +0.15),
    "Jurong East": (11, 7, 300, 15, 6800, 25, 110, 80, 3, 5, 3, +0.05, +0.05, +0.10, +0.06, +0.03),
    "Woodlands":   (14, 9, 260, 22, 5200, 32,  60, 72, 4, 4, 3, +0.30, +0.35, -0.12, -0.08, -0.10),
    "Bishan":      ( 9, 6, 240, 10, 5800, 24, 130, 75, 3, 4, 3, -0.05, -0.08, +0.12, +0.10, +0.08),
    "Bugis":       ( 8, 6, 220,  8, 6200, 18, 200, 85, 2, 4, 4, -0.08, -0.10, +0.18, +0.20, +0.05),
    "Toa Payoh":   (15, 9, 200, 25, 4800, 38,  45, 68, 3, 4, 3, +0.28, +0.32, -0.10, -0.05, -0.12),
    "Clementi":    (11, 7, 215, 14, 5100, 30,  55, 73, 3, 4, 3, +0.08, +0.10, -0.05, +0.02, +0.02),
}

_HIGHLIGHT_METRICS = [
    "avg_wait_time", "missed_queue", "total_transactions",
    "counter_utilization", "corporate_clients",
]


@st.cache_data
def generate_data() -> pd.DataFrame:
    np.random.seed(42)
    months = pd.date_range("2024-01-01", "2026-03-01", freq="MS")
    n = len(months)
    records = []

    for branch, p in _PARAMS.items():
        w0, h0, tok0, mis0, txn0, sen0, cor0, ut0, sl0, sp0, ma0 = p[:11]
        tw, tm, tt, tc, tu = p[11:]

        for i, month in enumerate(months):
            t = i / (n - 1)
            season = 1 + 0.07 * np.sin(2 * np.pi * (month.month - 3) / 12)
            rn = lambda s: np.random.normal(0, s)

            records.append({
                "branch": branch,
                "month": month,
                "avg_wait_time":       max(3.0, round(w0   * (1 + tw * t) * season + rn(0.5), 1)),
                "avg_handling_time":   max(3.0, round(h0   * (1 + 0.02*t) * season + rn(0.3), 1)),
                "queue_tokens":        max(50,  int(tok0   * (1 + 0.08*t) * season + rn(15))),
                "missed_queue":        max(0,   int(mis0   * (1 + tm * t) + rn(2))),
                "total_transactions":  max(500, int(txn0   * (1 + tt * t) * season + rn(200))),
                "senior_pct":          round(min(60, max(10, sen0 * (1 + 0.015*t) + rn(1.0))), 1),
                "corporate_clients":   max(10,  int(cor0   * (1 + tc * t) * season + rn(10))),
                "counter_utilization": round(min(98, max(40, ut0 * (1 + tu * t) + rn(2.0))), 1),
                "staff_seedling":      max(1, int(sl0 + rn(0.4))),
                "staff_sapling":       max(1, int(sp0 + rn(0.3))),
                "staff_mature":        max(1, int(ma0 + rn(0.2))),
                "retail_customers":    max(200, int(txn0 * 0.85 * (1 + tt * t) * season + rn(150))),
            })

    return pd.DataFrame(records)


@st.cache_data
def compute_highlights(df: pd.DataFrame):
    latest = df["month"].max()
    recent = df[df["month"] >= latest - pd.DateOffset(months=2)]
    prior  = df[(df["month"] <  latest - pd.DateOffset(months=2)) &
                (df["month"] >= latest - pd.DateOffset(months=5))]

    rows = []
    for branch in BRANCHES:
        for metric in _HIGHLIGHT_METRICS:
            meta = METRIC_META[metric]
            r_val = recent[recent["branch"] == branch][metric].mean()
            p_val = prior[prior["branch"]  == branch][metric].mean()
            if p_val == 0:
                continue
            pct = (r_val - p_val) / p_val * 100
            lib = meta["lower_is_better"]
            good_score = -pct if lib else pct
            rows.append({
                "branch": branch, "metric": metric,
                "label": meta["label"], "unit": meta["unit"],
                "pct_change": pct, "current": r_val,
                "lower_is_better": lib, "good_score": good_score,
            })

    df_i = pd.DataFrame(rows)
    positives = df_i[df_i["good_score"] > 0].nlargest(3, "good_score").to_dict("records")
    negatives = df_i[df_i["good_score"] < 0].nsmallest(3, "good_score").to_dict("records")
    return positives, negatives
