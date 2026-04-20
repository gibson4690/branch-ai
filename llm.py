import os
import json
from pathlib import Path
import pandas as pd
from data import METRIC_META, BRANCHES, PLOT_CATALOGUE

_SETTINGS_LOCAL = Path(__file__).parent / ".claude" / "settings.local.json"

_CHARTABLE = [k for k in METRIC_META if k not in ("staff_seedling", "staff_sapling", "staff_mature")]

TOOLS = [
    {
        "name": "generate_plot",
        "description": (
            "Render a chart to visualise a key data finding. "
            f"Plot catalogue — {'; '.join(f'{k}: {v}' for k, v in PLOT_CATALOGUE.items())}. "
            "Call this tool up to 2 times for the most insightful charts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": list(PLOT_CATALOGUE),
                    "description": "Chart type from the plot catalogue",
                },
                "metric": {
                    "type": "string",
                    "enum": _CHARTABLE,
                    "description": "Metric column to visualise",
                },
                "branches": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Branch names to include. Pass [] for all 8 branches.",
                },
                "title": {"type": "string", "description": "Descriptive chart title"},
            },
            "required": ["type", "metric", "title"],
        },
    },
    {
        "name": "suggest_followup",
        "description": "Suggest exactly 3 concise follow-up questions for the General Manager to explore next.",
        "input_schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3 short, specific follow-up questions (max 12 words each)",
                }
            },
            "required": ["questions"],
        },
    },
]


def _get_api_key() -> str:
    try:
        data = json.loads(_SETTINGS_LOCAL.read_text())
        key = data.get("env", {}).get("ANTHROPIC_API_KEY", "")
        if key and key != "your-api-key-here":
            return key
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def build_prompt(question: str, df: pd.DataFrame, ctx: dict = None) -> str:
    latest_m = df["month"].max()
    recent = df[df["month"] >= latest_m - pd.DateOffset(months=2)]
    summary = recent.groupby("branch")[[
        "avg_wait_time", "missed_queue", "total_transactions",
        "counter_utilization", "corporate_clients", "senior_pct",
        "staff_seedling", "staff_sapling", "staff_mature",
    ]].mean().round(1)

    focus = ""
    if ctx and ctx.get("branch") and ctx.get("metric"):
        b, m = ctx["branch"], ctx["metric"]
        unit  = METRIC_META.get(m, {}).get("unit", "")
        label = METRIC_META.get(m, {}).get("label", m)
        vals  = df[df["branch"] == b].sort_values("month")[m].tolist()
        focus = (
            f"\nFocus: {label} at {b} branch.\n"
            f"27-month history (Jan 2024 – Mar 2026): {[round(v, 1) for v in vals]} {unit}\n"
        )

    return f"""You are a senior analytics assistant for UOSB, a local Singapore bank with 8 branches.
The Country Manager asks: "{question}"
{focus}
Branch performance data (3-month average, most recent period):
{summary.to_string()}

Metric guide:
- avg_wait_time: customer wait time (min) — lower is better
- missed_queue: customers who left unserved — lower is better
- total_transactions: total volume processed — higher is better
- counter_utilization: % capacity used; 70–90% optimal
- corporate_clients: corporate visits/month — higher is better
- senior_pct: % customers aged 60+ (contextual)
- queue_tokens: queue numbers issued — proxy for demand
- avg_handling_time: transaction handling time (min)

Write a concise, professional analysis (under 180 words) for senior management using this structure:

**Key Finding:** 1-2 sentences summarising the most important insight.

**Analysis:** 2-3 sentences citing specific numbers and branch names.

**Recommendations:**
  - Recommendation 1
  - Recommendation 2
  - Recommendation 3

Use proper markdown indentation. Then call generate_plot up to 2 times for the most insightful charts, and call suggest_followup with 3 short follow-up questions."""
