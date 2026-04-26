"""V5 agent graph with budget-controlled loop, reviewer, and guardrails.

Graph:
  START → input_guardrail ─(pass)→ prepare_state → concierge → data_engineer → data_analyst → reviewer ─(sufficient)→ output_guardrail → END
                          ─(fail)→ END                                                          ↑          ─(more)──────→ concierge ────────┘
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from data import generate_data, METRIC_META
from llm import _get_api_key


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm(max_tokens: int = 512) -> ChatAnthropic:
    return ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=_get_api_key(), max_tokens=max_tokens)


# ── Schemas ───────────────────────────────────────────────────────────────────

class InputGuardrailResult(BaseModel):
    passed: bool = Field(description="True if the question is relevant to bank branch operations and contains no harmful content")
    reason: str = Field(description="If blocked, a brief explanation of why. Empty string when passed.")


class OutputGuardrailResult(BaseModel):
    passed: bool = Field(description="True if the output contains no personally identifiable information (PII)")
    reason: str = Field(description="If blocked, describe the PII detected. Empty string when passed.")


class InsightQuestion(BaseModel):
    insight_question: str = Field(description="A specific analytical question about bank branch performance, distinct from existing insights, that builds on prior findings")


class InsightAnswer(BaseModel):
    insight_answer: str = Field(description="A concise hypothetical data-driven answer to the insight question")


class DatasetSelection(BaseModel):
    datasets: list[str] = Field(description="Names of datasets relevant to the question. Choose any subset of: 'performance', 'queue', 'staff'")


class PlotInstruction(BaseModel):
    type: str = Field(description="Chart type. One of: 'bar', 'line', 'area', 'scatter', 'heatmap', 'ranking'")
    dataset: str = Field(description="Dataset to use. One of: 'performance', 'queue', 'staff'")
    metric: str = Field(description="Primary metric column name that exists in the chosen dataset")
    metric_y: str = Field(default="", description="Second metric column for scatter plots only. Leave empty for all other types.")
    branches: list[str] = Field(default_factory=list, description="Branch names to include. Empty list = all 8 branches.")
    title: str = Field(description="Descriptive chart title")
    months_back: int = Field(default=0, description="Restrict to last N months. 0 = all available data.")


class InsightPlots(BaseModel):
    plots: list[PlotInstruction] = Field(description="1 to 2 plot instructions that best visualise the insight finding")


class ReviewerOutput(BaseModel):
    short_conclusion: str = Field(description="A concise conclusion that directly answers the main question, synthesised from the insights")
    facts: list[str] = Field(description="Data-driven facts in bullet-point form. Each fact MUST contain at least one specific data point or statistic (a number, percentage, duration, count, or named value) quoted directly from the insight answers. Facts without a concrete figure are not allowed.")
    insights_sufficient: bool = Field(description="True if the executive summary comprehensively answers the main question and no further analysis is needed")
    reason: str = Field(description="Brief explanation of the sufficiency decision")


# ── State ─────────────────────────────────────────────────────────────────────

class V5State(TypedDict):
    question: str
    output_text: str
    datasets: dict          # {"performance": df, "queue": df, "staff": df}
    insights: list          # list of {"insight_question": str, "insight_answer": str}
    insight_data: list      # list of {"data_aggregations": [df, ...], "data_context": str}
    route: list             # ordered list of node names visited
    budget: int             # max number of engineer invocations
    spent_budget: int       # how many times concierge has routed to data_engineer
    executive_summary: dict  # {"short_conclusion": str, "facts": list[str]}
    reviewer_approved: bool  # set by reviewer; True = insights sufficient = go to output_guardrail
    reviewer_reason: str     # reviewer's explanation
    prepare_state_done: bool
    concierge_done: bool
    engineer_done: bool
    analyst_done: bool
    reviewer_done: bool
    input_guardrail_passed: bool
    input_guardrail_reason: str
    output_guardrail_passed: bool
    output_guardrail_reason: str


# ── Plot renderer ────────────────────────────────────────────────────────────

_LAYOUT = dict(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=0, r=0, t=36, b=0))

_DATASET_COLUMNS = {
    "performance": {"avg_wait_time", "avg_handling_time", "counter_utilization"},
    "queue":       {"queue_tokens", "missed_queue", "total_transactions"},
    "staff":       {"staff_seedling", "staff_sapling", "staff_mature", "senior_pct", "corporate_clients", "retail_customers"},
}


def generate_plot_from_instruction(instruction: dict, datasets: dict):
    """Render a Plotly figure from a dict-based plot instruction and a datasets dict."""
    chart_type   = instruction.get("type", "bar")
    dataset_name = instruction.get("dataset", "performance")
    metric       = instruction.get("metric")
    metric_y     = instruction.get("metric_y", "")
    branches     = instruction.get("branches") or []
    title        = instruction.get("title", "")
    months_back  = instruction.get("months_back", 0)

    if dataset_name not in datasets or not metric:
        return None
    df = datasets[dataset_name].copy()
    if metric not in df.columns:
        return None

    meta  = METRIC_META.get(metric, {})
    label = meta.get("label", metric)
    unit  = meta.get("unit", "")
    asc   = meta.get("lower_is_better", False)

    if months_back:
        cutoff = df["month"].max() - pd.DateOffset(months=int(months_back) - 1)
        df = df[df["month"] >= cutoff]
    subset = df[df["branch"].isin(branches)] if branches else df

    if chart_type == "line":
        fig = px.line(subset.sort_values("month"), x="month", y=metric, color="branch",
                      title=title or f"{label} — Monthly Trend",
                      labels={"month": "", metric: f"{label} ({unit})"})
        fig.update_layout(height=300, legend=dict(orientation="h", y=-0.28, font_size=11), **_LAYOUT)
        return fig

    if chart_type == "area":
        fig = px.area(subset.sort_values("month"), x="month", y=metric, color="branch",
                      title=title or f"{label} — Trend",
                      labels={"month": "", metric: f"{label} ({unit})"})
        fig.update_layout(height=300, legend=dict(orientation="h", y=-0.28, font_size=11), **_LAYOUT)
        return fig

    if chart_type == "bar":
        snapshot = subset[subset["month"] == subset["month"].max()].copy()
        fig = px.bar(snapshot.sort_values(metric, ascending=asc),
                     x="branch", y=metric,
                     title=title or f"{label} — Branch Comparison",
                     labels={"branch": "", metric: unit},
                     color=metric, color_continuous_scale="RdYlGn_r" if asc else "RdYlGn")
        fig.update_layout(height=280, coloraxis_showscale=False, xaxis_tickangle=-30, **_LAYOUT)
        return fig

    if chart_type == "ranking":
        agg = subset.groupby("branch")[metric].mean().reset_index()
        fig = px.bar(agg.sort_values(metric, ascending=not asc),
                     x=metric, y="branch", orientation="h",
                     title=title or f"{label} — Branch Ranking",
                     labels={"branch": "", metric: f"{label} ({unit})"},
                     color=metric, color_continuous_scale="RdYlGn_r" if asc else "RdYlGn")
        fig.update_layout(height=max(220, 36 * len(agg) + 60), coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), **_LAYOUT)
        return fig

    if chart_type == "scatter" and metric_y and metric_y in df.columns:
        agg = subset.groupby("branch")[[metric, metric_y]].mean().round(2).reset_index()
        meta_y  = METRIC_META.get(metric_y, {})
        label_y = meta_y.get("label", metric_y)
        unit_y  = meta_y.get("unit", "")
        fig = px.scatter(agg, x=metric, y=metric_y, text="branch",
                         title=title or f"{label} vs {label_y}",
                         labels={metric: f"{label} ({unit})", metric_y: f"{label_y} ({unit_y})"})
        fig.update_traces(textposition="top center")
        fig.update_layout(height=320, **_LAYOUT)
        return fig

    if chart_type == "heatmap":
        pivot = subset.pivot_table(index="branch", columns="month", values=metric, aggfunc="mean").round(2)
        pivot.columns = [c.strftime("%b %y") for c in pivot.columns]
        fig = px.imshow(pivot, title=title or f"{label} — Branch × Month",
                        color_continuous_scale="RdYlGn_r" if asc else "RdYlGn", aspect="auto")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=36, b=0), paper_bgcolor="white")
        return fig

    return None


# ── Guardrail nodes ───────────────────────────────────────────────────────────

def input_guardrail_node(state: V5State) -> V5State:
    question = state.get("question", "")
    prompt = (
        "You are a strict input guardrail for a bank branch analytics assistant.\n"
        "Evaluate whether the user question is relevant to bank branch operations, "
        "performance metrics, staff, queues, transactions, or customer insights, "
        "and whether it is free of harmful, violent, hateful, or inappropriate content.\n\n"
        f"Question: {question}"
    )
    result: InputGuardrailResult = _llm().with_structured_output(InputGuardrailResult).invoke(prompt)
    return {
        **state,
        "route": state.get("route", []) + ["input_guardrail"],
        "input_guardrail_passed": result.passed,
        "input_guardrail_reason": result.reason,
    }


def output_guardrail_node(state: V5State) -> V5State:
    text = state.get("output_text", "")
    prompt = (
        "You are a strict output guardrail for a bank branch analytics assistant.\n"
        "Scan the text below for personally identifiable information (PII) such as "
        "full names tied to records, NRIC/passport numbers, phone numbers, email addresses, "
        "home addresses, or any other sensitive personal details.\n\n"
        f"Text: {text}"
    )
    result: OutputGuardrailResult = _llm().with_structured_output(OutputGuardrailResult).invoke(prompt)
    return {
        **state,
        "route": state.get("route", []) + ["output_guardrail"],
        "output_guardrail_passed": result.passed,
        "output_guardrail_reason": result.reason,
    }


# ── Agent nodes ───────────────────────────────────────────────────────────────

def prepare_state_node(state: V5State) -> V5State:
    """Loads branch data into three thematic DataFrames and stores them in state."""
    df = generate_data()

    performance_cols = ["branch", "month", "avg_wait_time", "avg_handling_time", "counter_utilization"]
    queue_cols       = ["branch", "month", "queue_tokens", "missed_queue", "total_transactions"]
    staff_cols       = ["branch", "month", "staff_seedling", "staff_sapling", "staff_mature",
                        "senior_pct", "corporate_clients", "retail_customers"]

    datasets = {
        "performance": df[performance_cols].copy(),
        "queue":       df[queue_cols].copy(),
        "staff":       df[staff_cols].copy(),
    }

    return {
        **state,
        "route": state.get("route", []) + ["prepare_state"],
        "prepare_state_done": True,
        "datasets": datasets,
    }


def concierge_node(state: V5State) -> V5State:
    """Proposes the next insight question and always routes to data_engineer."""
    main_q = state.get("question", "")
    insights = list(state.get("insights", []))
    existing_insights = "\n".join(
        f"- Q: {i['insight_question']}\n  A: {i['insight_answer'] or '_pending_'}"
        for i in insights
    ) or "None"
    prompt = (
        f"You are a strict data concierge for a bank branch analytics assistant.\n"
        f"Main question: {main_q}\n"
        f"Insights explored so far:\n{existing_insights}\n\n"
        "Rules:\n"
        "1. If there are no insights yet AND the main question is a direct, data-answerable question "
        "(e.g. 'which branch has the longest wait time?'), use the main question itself as the insight question.\n"
        "2. Otherwise, propose the single most important question that is still unanswered and is "
        "DIRECTLY needed to answer the main question — not tangential analysis.\n"
        "3. Do NOT propose questions about topics already covered in existing insights.\n"
        "4. Do NOT add scope beyond what the main question asks for.\n"
        "5. The question must be answerable with branch performance data (wait times, queues, "
        "transactions, staff, utilisation).\n\n"
        "Propose the next insight question."
    )
    result: InsightQuestion = _llm().with_structured_output(InsightQuestion).invoke(prompt)
    insights = insights + [{"insight_question": result.insight_question, "insight_answer": ""}]

    return {
        **state,
        "route": state.get("route", []) + ["concierge"],
        "concierge_done": True,
        "spent_budget": state.get("spent_budget", 0) + 1,
        "insights": insights,
    }


def data_engineer_node(state: V5State) -> V5State:
    """Selects relevant datasets for the latest insight question, aggregates them, and builds data_context."""
    insights = state.get("insights", [])
    datasets = state.get("datasets", {})
    insight_data = list(state.get("insight_data", []))

    if not insights or not datasets:
        return {**state, "route": state.get("route", []) + ["data_engineer"], "engineer_done": True, "insight_data": insight_data}

    latest_question = insights[-1]["insight_question"]

    selection_prompt = (
        "You are a data engineer for a bank branch analytics assistant.\n"
        f"Insight question: {latest_question}\n\n"
        "Available datasets:\n"
        "- 'performance': avg_wait_time, avg_handling_time, counter_utilization\n"
        "- 'queue': queue_tokens, missed_queue, total_transactions\n"
        "- 'staff': staff_seedling, staff_sapling, staff_mature, senior_pct, corporate_clients, retail_customers\n\n"
        "Select only the datasets whose columns are directly needed to answer this question."
    )
    selection: DatasetSelection = _llm().with_structured_output(DatasetSelection).invoke(selection_prompt)
    selected_names = [k for k in selection.datasets if k in datasets] or list(datasets.keys())

    # Aggregate each selected dataset: 6-month average by branch
    data_aggregations = []
    markdown_parts = []
    for name in selected_names:
        df = datasets[name]
        latest_month = df["month"].max()
        recent = df[df["month"] >= latest_month - pd.DateOffset(months=5)]
        agg = recent.groupby("branch").mean(numeric_only=True).round(2).reset_index()
        data_aggregations.append(agg)
        markdown_parts.append(f"### {name.title()} (6-month avg by branch)\n\n{agg.to_markdown(index=False)}")

    data_context = "\n\n".join(markdown_parts)
    insight_data.append({"data_aggregations": data_aggregations, "data_context": data_context})

    return {
        **state,
        "route": state.get("route", []) + ["data_engineer"],
        "engineer_done": True,
        "insight_data": insight_data,
    }


def data_analyst_node(state: V5State) -> V5State:
    insights = list(state.get("insights", []))
    insight_data = state.get("insight_data", [])

    if insights and not insights[-1]["insight_answer"]:
        data_context = insight_data[-1]["data_context"] if insight_data else ""
        answer_prompt = (
            f"You are a data analyst for a bank branch analytics assistant.\n"
            f"Answer the insight question below using ONLY the data provided. "
            f"Cite specific figures from the data (branch names, numbers, percentages). "
            f"Be concise and direct.\n\n"
            f"Insight question: {insights[-1]['insight_question']}\n\n"
            f"Data:\n{data_context}"
        )
        answer: InsightAnswer = _llm(max_tokens=1024).with_structured_output(InsightAnswer).invoke(answer_prompt)

        # Generate plot instructions based on the question and answer
        available = "\n".join(
            f"- '{name}': {', '.join(sorted(cols))}"
            for name, cols in _DATASET_COLUMNS.items()
        )
        plot_prompt = (
            "You are a data visualisation expert for a bank branch analytics assistant.\n"
            f"Insight question: {insights[-1]['insight_question']}\n"
            f"Insight answer: {answer.insight_answer}\n\n"
            "Suggest 1 to 2 plots that best visualise this finding.\n\n"
            f"Available datasets and columns:\n{available}\n\n"
            "Chart types:\n"
            "- 'bar': branch comparison at the latest month\n"
            "- 'line': metric trend over time coloured by branch\n"
            "- 'area': filled volume trend over time\n"
            "- 'scatter': correlation between two metrics (set metric_y)\n"
            "- 'heatmap': branch × month intensity grid\n"
            "- 'ranking': horizontal bar ranking all branches by average\n\n"
            "Only use metric names that exist in the chosen dataset's column list above."
        )
        plots_result: InsightPlots = _llm(max_tokens=512).with_structured_output(InsightPlots).invoke(plot_prompt)
        plots = [p.model_dump() for p in plots_result.plots]

        insights[-1] = {**insights[-1], "insight_answer": answer.insight_answer, "plots": plots}

    output_text = insights[-1]["insight_answer"] if insights else ""
    return {
        **state,
        "route": state.get("route", []) + ["data_analyst"],
        "analyst_done": True,
        "insights": insights,
        "output_text": output_text,
    }


def reviewer_node(state: V5State) -> V5State:
    """Generates executive summary from insights, then decides sufficiency. Budget cap enforced here."""
    spent = state.get("spent_budget", 0)
    budget = state.get("budget", 1)
    insights = state.get("insights", [])

    existing_insights = "\n".join(
        f"- Q: {i['insight_question']}\n  A: {i['insight_answer'] or '_pending_'}"
        for i in insights
    ) or "None"

    prompt = (
        f"You are a strict senior reviewer for a bank branch analytics assistant.\n"
        f"Main question: {state.get('question', '')}\n"
        f"Insights gathered so far:\n{existing_insights}\n\n"
        "Step 1 — Write an executive summary:\n"
        "  - short_conclusion: one or two sentences that DIRECTLY answer the main question "
        "using specific figures from the insights. Do not be vague.\n"
        "  - facts: bullet points of data-driven facts from the insights that are DIRECTLY "
        "relevant to answering the main question. Exclude tangential findings. "
        "Every fact MUST quote at least one specific data point or statistic (a number, "
        "percentage, duration, count, or named value) from the insight answers — "
        "facts without a concrete figure must be discarded.\n\n"
        "Step 2 — Decide sufficiency:\n"
        "  Set insights_sufficient=True ONLY IF the executive summary above directly and "
        "specifically answers the main question with concrete data. "
        "Set it to False if the answer is still vague, incomplete, or missing key figures.\n"
        "  reason: one sentence explaining your decision."
    )
    output: ReviewerOutput = _llm(max_tokens=1024).with_structured_output(ReviewerOutput).invoke(prompt)

    # Hard budget cap overrides LLM decision
    approved = output.insights_sufficient or (spent >= budget)
    reason = f"Budget exhausted ({spent}/{budget} iterations used)." if spent >= budget else output.reason

    return {
        **state,
        "route": state.get("route", []) + ["reviewer"],
        "reviewer_done": True,
        "reviewer_approved": approved,
        "reviewer_reason": reason,
        "executive_summary": {
            "short_conclusion": output.short_conclusion,
            "facts": output.facts,
        },
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_input_guardrail(state: V5State) -> str:
    return "prepare_state" if state.get("input_guardrail_passed") else END


def _route_reviewer(state: V5State) -> str:
    return "output_guardrail" if state.get("reviewer_approved") else "concierge"


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(V5State)

    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("prepare_state", prepare_state_node)
    g.add_node("concierge", concierge_node)
    g.add_node("data_engineer", data_engineer_node)
    g.add_node("data_analyst", data_analyst_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("output_guardrail", output_guardrail_node)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", _route_input_guardrail, ["prepare_state", END])
    g.add_edge("prepare_state", "concierge")
    g.add_edge("concierge", "data_engineer")
    g.add_edge("data_engineer", "data_analyst")
    g.add_edge("data_analyst", "reviewer")
    g.add_conditional_edges("reviewer", _route_reviewer, ["concierge", "output_guardrail"])
    g.add_edge("output_guardrail", END)

    return g.compile()


graph_v5 = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_analysis_v5(question: str, *_args, budget: int = 3, **_kwargs) -> dict:
    """Run the V5 budget-controlled pipeline. budget sets max engineer invocations."""
    initial: V5State = {
        "question": question,
        "output_text": "",
        "datasets": {},
        "insights": [],
        "insight_data": [],
        "route": [],
        "budget": budget,
        "spent_budget": 0,
        "executive_summary": {},
        "reviewer_approved": False,
        "reviewer_reason": "",
        "prepare_state_done": False,
        "concierge_done": False,
        "engineer_done": False,
        "analyst_done": False,
        "reviewer_done": False,
        "input_guardrail_passed": False,
        "input_guardrail_reason": "",
        "output_guardrail_passed": False,
        "output_guardrail_reason": "",
    }

    s = graph_v5.invoke(initial)

    skipped = "— skipped"
    inp_blocked = not s["input_guardrail_passed"]
    rows = [
        ("Input Guardrail",  "✓ Pass" if s["input_guardrail_passed"] else f"✗ Blocked — {s['input_guardrail_reason']}"),
        ("Prepare State",    skipped if inp_blocked else ("✓" if s["prepare_state_done"] else "✗")),
        ("Concierge",        skipped if inp_blocked else f"✓ ({s['spent_budget']}/{s['budget']} budget used)"),
        ("Data Engineer",    skipped if inp_blocked else ("✓" if s["engineer_done"] else "✗")),
        ("Data Analyst",     skipped if inp_blocked else ("✓" if s["analyst_done"] else "✗")),
        ("Reviewer",         skipped if inp_blocked else (f"✓ Approved — {s['reviewer_reason']}" if s["reviewer_approved"] else f"↻ Loop — {s['reviewer_reason']}")),
        ("Output Guardrail", skipped if inp_blocked else ("✓ Pass" if s["output_guardrail_passed"] else f"✗ Blocked — {s['output_guardrail_reason']}")),
    ]
    table = "\n".join(f"| {agent} | {status} |" for agent, status in rows)

    route_lines = "\n".join(
        f"{i + 1}. `{node}`" for i, node in enumerate(s.get("route", []))
    )

    exec_summary = s.get("executive_summary", {})
    exec_md = ""
    if exec_summary:
        facts_md = "\n".join(f"- {f}" for f in exec_summary.get("facts", []))
        exec_md = (
            "**Executive Summary**\n\n"
            f"{exec_summary.get('short_conclusion', '')}\n\n"
            f"{facts_md}\n\n"
        )

    insights_md = ""
    for idx, ins in enumerate(s.get("insights", []), 1):
        insights_md += (
            f"**Insight {idx}**\n\n"
            f"- **Q:** {ins['insight_question']}\n"
            f"- **A:** {ins['insight_answer'] or '_pending_'}\n\n"
        )

    analysis = (
        "**V5 Agent Pipeline — Final State**\n\n"
        "**Route taken:**\n\n"
        f"{route_lines}\n\n"
        "| Agent | Status |\n"
        "|---|---|\n"
        f"{table}\n\n"
        + exec_md
        + (("**Insights:**\n\n" + insights_md) if insights_md else "")
    )

    insight_plots = [ins.get("plots", []) for ins in s.get("insights", [])]

    return {"analysis": analysis, "charts": {}, "follow_up": [], "insight_plots": insight_plots}
