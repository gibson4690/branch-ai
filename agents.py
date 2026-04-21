"""LangGraph ReAct agent for Branch.ai deep dive analysis."""
from __future__ import annotations

import json
from typing import Annotated, Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from data import METRIC_META, BRANCHES, PLOT_CATALOGUE
from llm import _get_api_key

# ── State ──────────────────────────────────────────────────────────────────────

class AnalysisState(TypedDict):
    question: str
    context: dict                          # branch/metric from card click
    data_summary: str                      # pre-formatted for the agent
    messages: Annotated[list, add_messages]
    analysis_text: str
    charts: list
    follow_up: list


# ── Tools ──────────────────────────────────────────────────────────────────────

_df: pd.DataFrame | None = None  # injected before graph runs

_CHARTABLE = [k for k in METRIC_META if k not in ("staff_seedling", "staff_sapling", "staff_mature")]


@tool
def query_data(
    metric: str,
    branches: list[str] | None = None,
    months_back: int = 3,
) -> str:
    """Query branch performance data. Returns a formatted table.

    Args:
        metric: One of the available metric column names.
        branches: Branch names to include. Pass [] or null for all branches.
        months_back: How many months of recent data to summarise (1-27).
    """
    if _df is None:
        return "Data not available."
    if metric not in _df.columns:
        available = ", ".join(METRIC_META.keys())
        return f"Unknown metric '{metric}'. Available: {available}"

    latest = _df["month"].max()
    cutoff = latest - pd.DateOffset(months=max(1, months_back) - 1)
    subset = _df[_df["month"] >= cutoff]

    if branches:
        valid = [b for b in branches if b in BRANCHES]
        if valid:
            subset = subset[subset["branch"].isin(valid)]

    summary = (
        subset.groupby("branch")[metric]
        .agg(["mean", "min", "max"])
        .round(2)
        .rename(columns={"mean": "avg", "min": "min", "max": "max"})
        .sort_values("avg", ascending=METRIC_META.get(metric, {}).get("lower_is_better", True))
    )
    meta = METRIC_META.get(metric, {})
    header = f"{meta.get('label', metric)} ({meta.get('unit', '')}) — last {months_back} month(s):\n"
    return header + summary.to_string()


@tool
def generate_plot(
    type: str,
    metric: str,
    title: str,
    branches: list[str] | None = None,
    metric_y: str | None = None,
    months_back: int | None = None,
) -> str:
    """Request a chart to be rendered. Call up to 4 times for the most insightful charts.

    Chart types:
      line    — monthly trend over time for one metric
      bar     — branch comparison at the latest period
      area    — filled area trend; good for volume/demand metrics
      scatter — correlation between metric (x-axis) and metric_y (y-axis) across branches
      heatmap — branch × month intensity grid; reveals patterns across time
      ranking — horizontal bar ranking all branches for one metric

    Args:
        type: Chart type from the catalogue above.
        metric: Primary metric column (x-axis for scatter).
        title: Descriptive chart title.
        branches: Branch names to include. Pass [] for all branches.
        metric_y: Second metric for scatter plots (y-axis). Required for type=scatter.
        months_back: Restrict to this many recent months (matches query_data window). None = full history.
    """
    if metric not in _CHARTABLE:
        return f"Cannot plot '{metric}'. Choose from: {', '.join(_CHARTABLE)}"
    if type == "scatter" and metric_y and metric_y not in _CHARTABLE:
        return f"Cannot plot '{metric_y}'. Choose from: {', '.join(_CHARTABLE)}"
    return json.dumps({
        "type": type,
        "metric": metric,
        "title": title,
        "branches": branches or [],
        "metric_y": metric_y,
        "months_back": months_back,
    })


@tool
def suggest_followup(questions: list[str]) -> str:
    """Emit exactly 3 short follow-up questions for the Country Manager.

    Args:
        questions: List of exactly 3 concise follow-up questions (max 12 words each).
    """
    return json.dumps({"questions": questions[:3]})


TOOLS = [query_data, generate_plot, suggest_followup]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── Nodes ──────────────────────────────────────────────────────────────────────

def prepare_context(state: AnalysisState) -> dict:
    """Build a compact data summary and prime the message list."""
    df = _df
    latest_m = df["month"].max()
    recent = df[df["month"] >= latest_m - pd.DateOffset(months=2)]
    summary = (
        recent.groupby("branch")[[
            "avg_wait_time", "missed_queue", "total_transactions",
            "counter_utilization", "corporate_clients", "senior_pct",
        ]]
        .mean()
        .round(1)
        .to_string()
    )

    ctx = state.get("context") or {}
    focus = ""
    if ctx.get("branch") and ctx.get("metric"):
        b, m = ctx["branch"], ctx["metric"]
        meta = METRIC_META.get(m, {})
        vals = df[df["branch"] == b].sort_values("month")[m].tolist()
        focus = (
            f"\nFocus: {meta.get('label', m)} at {b} branch. "
            f"27-month history: {[round(v, 1) for v in vals]} {meta.get('unit', '')}\n"
        )

    system = (
        "You are a senior analytics assistant for UOSB, a local Singapore bank with 8 branches "
        "(Orchard, Tampines, Jurong East, Woodlands, Bishan, Bugis, Toa Payoh, Clementi). "
        "Data spans Jan 2024 – Mar 2026.\n\n"
        "Metric guide:\n"
        "- avg_wait_time: wait time (min) — lower is better\n"
        "- missed_queue: customers who left unserved — lower is better\n"
        "- total_transactions: volume — higher is better\n"
        "- counter_utilization: % capacity; 70-90% optimal\n"
        "- corporate_clients: corporate visits/month — higher is better\n"
        "- senior_pct: % customers aged 60+ (contextual)\n"
        "- queue_tokens: demand proxy\n"
        "- avg_handling_time: handling time (min)\n\n"
        "Staff seniority levels: seedling (junior), sapling (mid), mature (senior).\n\n"
        "Recent 3-month branch averages:\n"
        f"{summary}\n{focus}\n"
        "Write a concise professional analysis for senior management using this structure:\n\n"
        "**Key Finding:** 1-2 sentences.\n\n"
        "**Analysis:** 2-3 sentences with specific numbers and branch names.\n\n"
        "**Recommendations:**\n"
        "  - Recommendation 1\n"
        "  - Recommendation 2\n"
        "  - Recommendation 3\n\n"
        "Use query_data to fetch the specific data you need before forming conclusions.\n"
        "Call generate_plot 2–4 times to visually support your findings — choose chart types that best "
        "illustrate the analysis: use scatter to show correlations between metrics, heatmap to show "
        "patterns across branches and time, area for volume trends, ranking to compare all branches. "
        "Match the branches and months_back to the data you actually queried. "
        "Always end by calling suggest_followup with 3 short follow-up questions."
    )

    return {
        "data_summary": summary,
        "messages": [
            HumanMessage(content=f"System context:\n{system}\n\nQuestion: {state['question']}")
        ],
        "charts": [],
        "follow_up": [],
        "analysis_text": "",
    }


def analyst(state: AnalysisState) -> dict:
    """ReAct agent node — calls LLM and collects tool calls."""
    api_key = _get_api_key()
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=api_key,
        max_tokens=1200,
    ).bind_tools(TOOLS)

    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_executor(state: AnalysisState) -> dict:
    """Execute all tool calls from the last AI message."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}

    tool_messages = []
    new_charts = list(state.get("charts", []))
    new_followup = list(state.get("follow_up", []))

    for tc in last.tool_calls:
        tool_fn = TOOL_MAP.get(tc["name"])
        if tool_fn is None:
            result = f"Unknown tool: {tc['name']}"
        else:
            try:
                result = tool_fn.invoke(tc["args"])
            except Exception as e:
                result = f"Tool error: {e}"

        # Capture chart specs and follow-up questions
        if tc["name"] == "generate_plot":
            try:
                spec = json.loads(result) if isinstance(result, str) else result
                new_charts.append(spec)
            except (json.JSONDecodeError, TypeError):
                pass
        elif tc["name"] == "suggest_followup":
            try:
                data = json.loads(result) if isinstance(result, str) else result
                new_followup = data.get("questions", [])[:3]
            except (json.JSONDecodeError, TypeError):
                pass

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages, "charts": new_charts, "follow_up": new_followup}


def extract_text(state: AnalysisState) -> dict:
    """Pull the final analysis text from AI messages."""
    text_parts = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
    return {"analysis_text": "\n\n".join(t for t in text_parts if t.strip())}


# ── Routing ────────────────────────────────────────────────────────────────────

def should_continue(state: AnalysisState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        # If suggest_followup was called, we're done after tool execution
        tool_names = {tc["name"] for tc in last.tool_calls}
        if "suggest_followup" in tool_names:
            return "tools_then_end"
        return "tools"
    return "end"


# ── Graph ──────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AnalysisState)

    g.add_node("prepare_context", prepare_context)
    g.add_node("analyst", analyst)
    g.add_node("tool_executor", tool_executor)
    g.add_node("extract_text", extract_text)

    g.add_edge(START, "prepare_context")
    g.add_edge("prepare_context", "analyst")

    g.add_conditional_edges(
        "analyst",
        should_continue,
        {
            "tools": "tool_executor",
            "tools_then_end": "tool_executor",
            "end": "extract_text",
        },
    )

    g.add_edge("tool_executor", "analyst")
    g.add_edge("extract_text", END)

    return g.compile()


graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_analysis(question: str, df: pd.DataFrame, ctx: dict | None = None) -> dict:
    """Run the analysis graph synchronously. Returns {analysis, charts, follow_up}."""
    global _df
    _df = df

    result = graph.invoke({
        "question": question,
        "context": ctx or {},
        "messages": [],
        "data_summary": "",
        "analysis_text": "",
        "charts": [],
        "follow_up": [],
    })

    return {
        "analysis": result.get("analysis_text", ""),
        "charts": result.get("charts", []),
        "follow_up": result.get("follow_up", []),
    }
