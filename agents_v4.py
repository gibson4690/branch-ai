"""V4 agent graph with budget-controlled loop, reviewer, and guardrails.

Graph:
  START → input_guardrail ─(pass)→ concierge → data_engineer → data_analyst → reviewer ─(sufficient)→ output_guardrail → END
                          ─(fail)→ END                                         ↑          ─(more)──────→ concierge ────────┘
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

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


class ReviewerOutput(BaseModel):
    short_conclusion: str = Field(description="A concise conclusion that directly answers the main question, synthesised from the insights")
    facts: list[str] = Field(description="Data-driven facts in bullet-point form. Each fact MUST contain at least one specific data point or statistic (a number, percentage, duration, count, or named value) quoted directly from the insight answers. Facts without a concrete figure are not allowed.")
    insights_sufficient: bool = Field(description="True if the executive summary comprehensively answers the main question and no further analysis is needed")
    reason: str = Field(description="Brief explanation of the sufficiency decision")


# ── State ─────────────────────────────────────────────────────────────────────

class V4State(TypedDict):
    question: str
    output_text: str
    insights: list          # list of {"insight_question": str, "insight_answer": str}
    route: list             # ordered list of node names visited
    budget: int             # max number of engineer invocations
    spent_budget: int       # how many times concierge has routed to data_engineer
    executive_summary: dict  # {"short_conclusion": str, "facts": list[str]}
    reviewer_approved: bool  # set by reviewer; True = insights sufficient = go to output_guardrail
    reviewer_reason: str     # reviewer's explanation
    concierge_done: bool
    engineer_done: bool
    analyst_done: bool
    reviewer_done: bool
    input_guardrail_passed: bool
    input_guardrail_reason: str
    output_guardrail_passed: bool
    output_guardrail_reason: str


# ── Guardrail nodes ───────────────────────────────────────────────────────────

def input_guardrail_node(state: V4State) -> V4State:
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


def output_guardrail_node(state: V4State) -> V4State:
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

def concierge_node(state: V4State) -> V4State:
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


def data_engineer_node(state: V4State) -> V4State:
    return {**state, "route": state.get("route", []) + ["data_engineer"], "engineer_done": True}


def data_analyst_node(state: V4State) -> V4State:
    insights = list(state.get("insights", []))

    if insights and not insights[-1]["insight_answer"]:
        prompt = (
            f"You are a data analyst for a bank branch analytics assistant.\n"
            f"Provide a concise hypothetical data-driven answer to this insight question "
            f"as if you had access to UOSB bank branch performance data (Jan 2024 – Mar 2026):\n\n"
            f"{insights[-1]['insight_question']}"
        )
        result: InsightAnswer = _llm(max_tokens=1024).with_structured_output(InsightAnswer).invoke(prompt)
        insights[-1] = {**insights[-1], "insight_answer": result.insight_answer}

    output_text = insights[-1]["insight_answer"] if insights else ""
    return {
        **state,
        "route": state.get("route", []) + ["data_analyst"],
        "analyst_done": True,
        "insights": insights,
        "output_text": output_text,
    }


def reviewer_node(state: V4State) -> V4State:
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

def _route_input_guardrail(state: V4State) -> str:
    return "concierge" if state.get("input_guardrail_passed") else END


def _route_reviewer(state: V4State) -> str:
    return "output_guardrail" if state.get("reviewer_approved") else "concierge"


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(V4State)

    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("concierge", concierge_node)
    g.add_node("data_engineer", data_engineer_node)
    g.add_node("data_analyst", data_analyst_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("output_guardrail", output_guardrail_node)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", _route_input_guardrail, ["concierge", END])
    g.add_edge("concierge", "data_engineer")
    g.add_edge("data_engineer", "data_analyst")
    g.add_edge("data_analyst", "reviewer")
    g.add_conditional_edges("reviewer", _route_reviewer, ["concierge", "output_guardrail"])
    g.add_edge("output_guardrail", END)

    return g.compile()


graph_v4 = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_analysis_v4(question: str, *_args, budget: int = 3, **_kwargs) -> dict:
    """Run the V4 budget-controlled pipeline. budget sets max engineer invocations."""
    initial: V4State = {
        "question": question,
        "output_text": "",
        "insights": [],
        "route": [],
        "budget": budget,
        "spent_budget": 0,
        "executive_summary": {},
        "reviewer_approved": False,
        "reviewer_reason": "",
        "concierge_done": False,
        "engineer_done": False,
        "analyst_done": False,
        "reviewer_done": False,
        "input_guardrail_passed": False,
        "input_guardrail_reason": "",
        "output_guardrail_passed": False,
        "output_guardrail_reason": "",
    }

    s = graph_v4.invoke(initial)

    skipped = "— skipped"
    inp_blocked = not s["input_guardrail_passed"]
    rows = [
        ("Input Guardrail",  "✓ Pass" if s["input_guardrail_passed"] else f"✗ Blocked — {s['input_guardrail_reason']}"),
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
        "**V4 Agent Pipeline — Final State**\n\n"
        "**Route taken:**\n\n"
        f"{route_lines}\n\n"
        "| Agent | Status |\n"
        "|---|---|\n"
        f"{table}\n\n"
        + exec_md
        + (("**Insights:**\n\n" + insights_md) if insights_md else "")
    )

    return {"analysis": analysis, "charts": {}, "follow_up": []}
