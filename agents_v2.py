"""Multi-agent LangGraph system implementing the AGENTS.md architecture.

Graph: START → DataConcierge → Reviewer ⟷ DataEngineer / DataAnalyst → Executive → END
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from data import METRIC_META, BRANCHES
from llm import _get_api_key

# ── Global dataframe (injected before graph runs) ─────────────────────────────
_df: pd.DataFrame | None = None
_CHARTABLE = [k for k in METRIC_META if k not in ("staff_seedling", "staff_sapling", "staff_mature")]


# ── State ─────────────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    business_question: str
    analysis_plan: list          # list of InsightPlan dicts
    executive_summary: str
    data_catalog: dict
    follow_up: list
    charts: dict                 # accumulated from all insights
    current_insight_idx: int


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_data_catalog() -> str:
    """Retrieve the data catalog describing all available datasets and their columns."""
    if _df is None:
        return "No data available."
    catalog = {
        "branch_performance": {
            "description": "Monthly branch performance metrics for UOSB bank (Jan 2024 – Mar 2026)",
            "columns": {
                col: {
                    "label": METRIC_META.get(col, {}).get("label", col),
                    "unit": METRIC_META.get(col, {}).get("unit", ""),
                    "lower_is_better": METRIC_META.get(col, {}).get("lower_is_better"),
                }
                for col in _df.columns if col not in ("branch", "month")
            },
            "branches": BRANCHES,
            "row_count": len(_df),
        }
    }
    return json.dumps(catalog, indent=2)


@tool
def query_data(
    metric: str,
    groupby: str = "branch",
    agg_func: str = "mean",
    filter_branches: list[str] | None = None,
    months_back: int | None = None,
    sort_ascending: bool = True,
) -> str:
    """Query branch performance data with flexible groupby and aggregation.

    Args:
        metric: Column to aggregate (e.g. avg_wait_time, missed_queue, total_transactions).
        groupby: Column(s) to group by: 'branch', 'month', or 'branch,month'.
        agg_func: Aggregation function: 'mean', 'sum', 'min', 'max', 'count'.
        filter_branches: List of branch names to include. None = all branches.
        months_back: Restrict to this many recent months. None = full history.
        sort_ascending: Sort result ascending (True) or descending (False).
    """
    if _df is None:
        return "Data not available."
    if metric not in _df.columns:
        return f"Unknown metric '{metric}'. Available: {', '.join(METRIC_META.keys())}"

    base = _df.copy()
    if months_back:
        latest = base["month"].max()
        cutoff = latest - pd.DateOffset(months=int(months_back) - 1)
        base = base[base["month"] >= cutoff]
    if filter_branches:
        valid = [b for b in filter_branches if b in BRANCHES]
        if valid:
            base = base[base["branch"].isin(valid)]

    gb_cols = [c.strip() for c in groupby.split(",") if c.strip() in base.columns]
    if not gb_cols:
        gb_cols = ["branch"]

    agg_fn = {"mean": "mean", "sum": "sum", "min": "min", "max": "max", "count": "count"}.get(agg_func, "mean")
    result = base.groupby(gb_cols)[metric].agg(agg_fn).reset_index()
    result[metric] = result[metric].round(2)
    result = result.sort_values(metric, ascending=sort_ascending)

    meta = METRIC_META.get(metric, {})
    header = f"{meta.get('label', metric)} ({meta.get('unit', '')}) — grouped by {groupby} [{agg_func}]:\n"
    return header + result.to_string(index=False)


@tool
def generate_chart(
    chart_id: str,
    type: str,
    metric: str,
    title: str,
    branches: list[str] | None = None,
    metric_y: str | None = None,
    months_back: int | None = None,
) -> str:
    """Request a chart to be rendered inline in the analysis.

    IMPORTANT: Place [CHART:chart_id] in your analysis text right after the sentence
    this chart supports.

    Chart types: line, bar, area, scatter, heatmap, ranking

    Args:
        chart_id: Unique ID e.g. "i1_chart1". Must match the [CHART:chart_id] marker.
        type: Chart type from the catalogue above.
        metric: Primary metric column.
        title: Descriptive chart title.
        branches: Branch names to include. Pass [] for all branches.
        metric_y: Second metric for scatter plots (y-axis).
        months_back: Restrict to this many recent months. None = full history.
    """
    if metric not in _CHARTABLE:
        return f"Cannot plot '{metric}'. Choose from: {', '.join(_CHARTABLE)}"
    return json.dumps({
        "chart_id": chart_id,
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


ENGINEER_TOOLS = [get_data_catalog, query_data]
ANALYST_TOOLS = [generate_chart]
EXECUTIVE_TOOLS = [suggest_followup]
ENGINEER_TOOL_MAP = {t.name: t for t in ENGINEER_TOOLS}
ANALYST_TOOL_MAP = {t.name: t for t in ANALYST_TOOLS}
EXECUTIVE_TOOL_MAP = {t.name: t for t in EXECUTIVE_TOOLS}


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm(max_tokens: int = 1200) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=_get_api_key(),
        max_tokens=max_tokens,
    )


def _run_react_loop(
    messages: list,
    tools: list,
    tool_map: dict,
    max_iters: int = 6,
) -> tuple[list, dict, dict]:
    """Run a ReAct tool loop. Returns (messages, charts, follow_up_dict)."""
    messages = list(messages)
    llm = _llm().bind_tools(tools)
    charts: dict = {}
    follow_up: dict = {}

    for _ in range(max_iters):
        response = llm.invoke(messages)
        messages.append(response)

        if not isinstance(response, AIMessage) or not response.tool_calls:
            break

        tool_messages = []
        should_stop = False

        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            try:
                result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
            except Exception as e:
                result = f"Tool error: {e}"

            try:
                if tc["name"] == "generate_chart":
                    spec = json.loads(result)
                    charts[spec["chart_id"]] = spec
                elif tc["name"] == "suggest_followup":
                    follow_up = json.loads(result)
                    should_stop = True
            except (json.JSONDecodeError, TypeError, KeyError):
                pass

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        messages.extend(tool_messages)

        if should_stop:
            break

    return messages, charts, follow_up


def _extract_text(messages: list) -> str:
    """Collect text content from all AI messages in a conversation."""
    parts = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block["text"])
    return "\n\n".join(p for p in parts if p.strip())


# ── Node: DataConciergeAgent ──────────────────────────────────────────────────

def data_concierge(state: MultiAgentState) -> dict:
    """Creates the Analysis Plan — a list of focused insight questions."""
    question = state["business_question"]

    prompt = (
        "You are a Data Concierge for UOSB bank in Singapore. "
        "Given a business question from a Country Manager, create an analysis plan "
        "with 2-3 specific insight questions that together fully answer the business question.\n\n"
        f"Available metrics: {', '.join(METRIC_META.keys())}.\n"
        f"Branches: {', '.join(BRANCHES)}.\n"
        "Data range: Jan 2024 – Mar 2026.\n\n"
        "Respond with ONLY a JSON array. Each element must have:\n"
        '  "insight_question": specific analytical question (string)\n'
        '  "data_query_requests": list of 1-3 strings describing what data to fetch\n\n'
        "Example:\n"
        '[\n'
        '  {"insight_question": "Which branches have the highest wait times?",\n'
        '   "data_query_requests": ["avg_wait_time by branch last 6 months", "monthly trend of avg_wait_time"]}\n'
        ']\n\n'
        f"Business question: {question}"
    )

    response = _llm(max_tokens=600).invoke([HumanMessage(content=prompt)])
    content = response.content if isinstance(response.content, str) else ""

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        plan_data = json.loads(content[start:end]) if start != -1 and end > start else []
    except (json.JSONDecodeError, ValueError):
        plan_data = []

    if not plan_data:
        plan_data = [{"insight_question": question, "data_query_requests": ["branch performance overview"]}]

    analysis_plan = [
        {
            "insight_question": item.get("insight_question", question),
            "insight_short_answer": "",
            "insight_long_answer": "",
            "insight_key_facts": [],
            "data_query_requests": item.get("data_query_requests", []),
            "insight_datasets": {},
            "insight_data_catalog": {},
            "analysis_text": "",
            "charts": {},
            "completed": False,
        }
        for item in plan_data[:3]
    ]

    return {
        "analysis_plan": analysis_plan,
        "data_catalog": {},
        "current_insight_idx": 0,
        "charts": {},
        "follow_up": [],
        "executive_summary": "",
    }


# ── Node: DataEngineerAgent ────────────────────────────────────────────────────

def data_engineer(state: MultiAgentState) -> dict:
    """Queries data for the current insight and builds its data dictionary."""
    idx = state["current_insight_idx"]
    plan = list(state["analysis_plan"])
    insight = dict(plan[idx])

    prompt = (
        "You are a Data Engineer for UOSB bank. "
        "Specify 2-3 data queries needed to answer this insight question.\n\n"
        f"Insight question: {insight['insight_question']}\n"
        f"Data needed: {', '.join(insight.get('data_query_requests', []))}\n\n"
        f"Available metrics: {', '.join(METRIC_META.keys())}\n\n"
        "Respond with ONLY a JSON array. Each element must have:\n"
        '  "metric": metric column name\n'
        '  "groupby": "branch", "month", or "branch,month"\n'
        '  "agg_func": "mean", "sum", "min", or "max"\n'
        '  "months_back": integer or null\n'
        '  "filter_branches": list of branch names or null\n'
        '  "sort_ascending": true or false'
    )

    response = _llm(max_tokens=400).invoke([HumanMessage(content=prompt)])
    content = response.content if isinstance(response.content, str) else ""

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        specs = json.loads(content[start:end]) if start != -1 and end > start else []
    except (json.JSONDecodeError, ValueError):
        specs = []

    datasets: dict = {}
    data_catalog: dict = {}

    for spec in specs[:3]:
        metric = spec.get("metric", "")
        if metric not in METRIC_META:
            continue
        groupby = spec.get("groupby", "branch")
        agg_func = spec.get("agg_func", "mean")
        months_back = spec.get("months_back")
        filter_branches = spec.get("filter_branches")
        sort_asc = spec.get("sort_ascending", True)

        try:
            result = query_data.invoke({
                "metric": metric,
                "groupby": groupby,
                "agg_func": agg_func,
                "filter_branches": filter_branches,
                "months_back": months_back,
                "sort_ascending": sort_asc,
            })
        except Exception as e:
            result = f"Query error: {e}"

        name = f"{metric}_by_{groupby.replace(',', '_')}_{agg_func}"
        datasets[name] = {"dataset_name": name, "data_table": result}
        data_catalog[name] = {
            "metric": metric,
            "groupby": groupby,
            "agg_func": agg_func,
            "description": f"{METRIC_META[metric]['label']} grouped by {groupby}",
        }

    # Fallback if no valid specs were generated
    if not datasets:
        metric = "avg_wait_time"
        result = query_data.invoke({"metric": metric, "groupby": "branch", "agg_func": "mean"})
        name = f"{metric}_by_branch_mean"
        datasets[name] = {"dataset_name": name, "data_table": result}
        data_catalog[name] = {"metric": metric, "description": "Avg wait time by branch"}

    insight["insight_datasets"] = datasets
    insight["insight_data_catalog"] = data_catalog
    plan[idx] = insight

    return {"analysis_plan": plan}


# ── Node: DataAnalystAgent ────────────────────────────────────────────────────

def data_analyst(state: MultiAgentState) -> dict:
    """Analyses queried data for the current insight and generates charts."""
    idx = state["current_insight_idx"]
    plan = list(state["analysis_plan"])
    insight = dict(plan[idx])

    datasets_text = "\n".join(
        f"### {name}\n{ds.get('data_table', '')}"
        for name, ds in insight.get("insight_datasets", {}).items()
    ) or "No pre-fetched data. Use general knowledge of UOSB branch performance."

    chart_prefix = f"i{idx + 1}"

    system = (
        "You are a senior Data Analyst for UOSB bank. "
        "Write a concise professional analysis for senior management to answer the insight question.\n\n"
        "Use this structure (no headers needed, just the content):\n"
        "**Key Finding:** 1-2 sentences.\n"
        "**Analysis:** 2-3 sentences with specific numbers and branch names.\n\n"
        f"Call generate_chart 1-2 times. Use chart IDs like '{chart_prefix}_chart1', '{chart_prefix}_chart2'. "
        "Place [CHART:chart_id] in your text immediately after the sentence each chart supports. "
        "Be specific, data-driven, and concise. No filler text."
    )

    msgs = [HumanMessage(content=(
        f"System: {system}\n\n"
        f"Insight question: {insight['insight_question']}\n\n"
        f"Available data:\n{datasets_text}"
    ))]

    msgs, charts, _ = _run_react_loop(msgs, ANALYST_TOOLS, ANALYST_TOOL_MAP, max_iters=4)
    analysis_text = _extract_text(msgs)

    insight["analysis_text"] = analysis_text
    insight["charts"] = charts
    insight["completed"] = True
    plan[idx] = insight

    all_charts = dict(state.get("charts", {}))
    all_charts.update(charts)

    return {
        "analysis_plan": plan,
        "current_insight_idx": idx + 1,
        "charts": all_charts,
    }


# ── Node: ExecutiveAgent ──────────────────────────────────────────────────────

def executive_agent(state: MultiAgentState) -> dict:
    """Creates an executive summary from all completed insights."""
    plan = state["analysis_plan"]
    question = state["business_question"]
    all_charts = state.get("charts", {})

    insights_text = ""
    for i, ins in enumerate(plan):
        insights_text += (
            f"\n## Insight {i + 1}: {ins['insight_question']}\n"
            f"{ins.get('analysis_text', '(no analysis)')}\n"
        )

    available_markers = ", ".join(f"[CHART:{cid}]" for cid in all_charts) or "none"

    system = (
        "You are an Executive Analyst for UOSB bank. "
        "Create a concise executive summary for the Country Manager based on the completed insights.\n\n"
        "**Executive Summary**\n"
        "1-2 sentences directly answering the business question.\n\n"
        "**Key Findings**\n"
        "- Finding 1 (with specific numbers and branch names)\n"
        "- Finding 2\n"
        "- Finding 3\n\n"
        "**Recommendations**\n"
        "- Recommendation 1\n"
        "- Recommendation 2\n"
        "- Recommendation 3\n\n"
        f"Available inline chart markers (use where relevant): {available_markers}\n\n"
        "After the summary, call suggest_followup with exactly 3 short follow-up questions. "
        "Keep the entire summary concise and professional."
    )

    msgs = [HumanMessage(content=(
        f"System: {system}\n\n"
        f"Business question: {question}\n\n"
        f"Completed insights:\n{insights_text}"
    ))]

    msgs, _, follow_up_data = _run_react_loop(msgs, EXECUTIVE_TOOLS, EXECUTIVE_TOOL_MAP, max_iters=3)
    summary = _extract_text(msgs)
    follow_up = follow_up_data.get("questions", [])[:3]

    return {"executive_summary": summary, "follow_up": follow_up}


# ── ReviewerAgent (routing logic) ─────────────────────────────────────────────

def reviewer_router(state: MultiAgentState) -> str:
    """Routes to DataEngineer, DataAnalyst, or ExecutiveAgent based on plan state."""
    plan = state.get("analysis_plan", [])
    idx = state.get("current_insight_idx", 0)

    if not plan or idx >= len(plan):
        return "executive"

    insight = plan[idx]
    if not insight.get("insight_datasets"):
        return "data_engineer"
    return "data_analyst"


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(MultiAgentState)

    g.add_node("data_concierge", data_concierge)
    g.add_node("data_engineer", data_engineer)
    g.add_node("data_analyst", data_analyst)
    g.add_node("executive", executive_agent)

    g.add_edge(START, "data_concierge")

    _routes = {"data_engineer": "data_engineer", "data_analyst": "data_analyst", "executive": "executive"}
    g.add_conditional_edges("data_concierge", reviewer_router, _routes)
    g.add_conditional_edges("data_engineer", reviewer_router, _routes)
    g.add_conditional_edges("data_analyst", reviewer_router, _routes)

    g.add_edge("executive", END)

    return g.compile()


graph_v2 = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_analysis_v2(question: str, df: pd.DataFrame, ctx: dict | None = None) -> dict:
    """Run the multi-agent analysis. Returns {analysis, charts, follow_up}."""
    global _df
    _df = df

    enhanced_question = question
    if ctx:
        branch = ctx.get("branch", "")
        metric = ctx.get("metric", "")
        if branch and metric:
            meta = METRIC_META.get(metric, {})
            enhanced_question = f"{question} (Focus: {meta.get('label', metric)} at {branch} branch)"

    result = graph_v2.invoke({
        "business_question": enhanced_question,
        "analysis_plan": [],
        "executive_summary": "",
        "data_catalog": {},
        "follow_up": [],
        "charts": {},
        "current_insight_idx": 0,
    })

    return {
        "analysis": result.get("executive_summary", ""),
        "charts": result.get("charts", {}),
        "follow_up": result.get("follow_up", []),
    }
