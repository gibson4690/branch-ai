import os
import json
import re
from pathlib import Path
import pandas as pd
from data import METRIC_META

_SETTINGS_LOCAL = Path(__file__).parent / ".claude" / "settings.local.json"


def _get_api_key() -> str:
    try:
        data = json.loads(_SETTINGS_LOCAL.read_text())
        key = data.get("env", {}).get("ANTHROPIC_API_KEY", "")
        if key and key != "your-api-key-here":
            return key
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def get_analysis(question: str, df: pd.DataFrame, ctx: dict = None) -> dict:
    """Returns {"analysis": str, "charts": list[dict]}"""
    try:
        import anthropic
    except ImportError:
        return {"analysis": "_LLM unavailable — install the `anthropic` package._", "charts": []}

    api_key = _get_api_key()
    if not api_key:
        return {
            "analysis": "_AI analysis requires `ANTHROPIC_API_KEY` to be set in `.claude/settings.local.json`._",
            "charts": [],
        }

    latest = df["month"].max()
    recent = df[df["month"] >= latest - pd.DateOffset(months=2)]
    summary = recent.groupby("branch")[[
        "avg_wait_time", "missed_queue", "total_transactions",
        "counter_utilization", "corporate_clients", "senior_pct",
        "staff_seedling", "staff_sapling", "staff_mature",
    ]].mean().round(1)

    focus = ""
    if ctx and ctx.get("branch") and ctx.get("metric"):
        branch, metric = ctx["branch"], ctx["metric"]
        unit  = METRIC_META.get(metric, {}).get("unit", "")
        label = METRIC_META.get(metric, {}).get("label", metric)
        vals  = df[df["branch"] == branch].sort_values("month")[metric].tolist()
        focus = (
            f"\nFocus: {label} at {branch} branch.\n"
            f"27-month history (Jan 2024–Mar 2026): {[round(v, 1) for v in vals]} {unit}\n"
        )

    available = [k for k in METRIC_META if k not in ("staff_seedling", "staff_sapling", "staff_mature")]

    prompt = f"""You are a banking analytics assistant for OCSG Bank, a Singapore retail bank with 8 branches.
The General Manager asks: "{question}"
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
- queue_tokens: queue numbers issued — higher means more demand
- avg_handling_time: transaction handling time (min)

Respond ONLY with a valid JSON object, no markdown, no extra text:
{{
  "analysis": "**Key Finding:** (1-2 sentences)\\n\\n**Analysis:** (2-3 sentences citing specific numbers)\\n\\n**Recommendations:**\\n• (action 1)\\n• (action 2)\\n• (action 3)",
  "charts": [
    {{
      "type": "line",
      "metric": "metric_name",
      "branches": null,
      "title": "Descriptive chart title"
    }},
    {{
      "type": "bar",
      "metric": "metric_name",
      "branches": null,
      "title": "Descriptive chart title"
    }}
  ]
}}

Rules:
- Provide exactly 2 charts that best visualise the key findings
- "type": "line" for monthly trends, "bar" for branch comparisons
- "branches": null for all 8 branches, or e.g. ["Orchard","Woodlands"] to focus
- Only use metrics from: {available}
- Analysis under 180 words, professional, data-driven, for senior management"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

        # Strip markdown code fence if present
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        text = m.group(1) if m else raw

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return {"analysis": raw, "charts": []}

        result.setdefault("analysis", raw)
        result.setdefault("charts", [])
        return result

    except Exception as e:
        return {"analysis": f"_Analysis error: {e}_", "charts": []}
