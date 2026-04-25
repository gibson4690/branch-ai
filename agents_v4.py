"""V4 agent graph with budget-controlled concierge loop and guardrails.

Graph:
  START → input_guardrail ─(pass)→ concierge ─(spent < budget)→ data_engineer → data_analyst ─┐
                          ─(fail)→ END         ─(spent == budget)→ output_guardrail → END       │
                                               └──────────────────────────────────────────────────┘
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from llm import _get_api_key


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=_get_api_key(), max_tokens=256)


# ── Guardrail schemas ─────────────────────────────────────────────────────────

class InputGuardrailResult(BaseModel):
    passed: bool = Field(description="True if the question is relevant to bank branch operations and contains no harmful content")
    reason: str = Field(description="If blocked, a brief explanation of why. Empty string when passed.")


class OutputGuardrailResult(BaseModel):
    passed: bool = Field(description="True if the output contains no personally identifiable information (PII)")
    reason: str = Field(description="If blocked, describe the PII detected. Empty string when passed.")


# ── State ─────────────────────────────────────────────────────────────────────

class V4State(TypedDict):
    question: str
    output_text: str
    route: list           # ordered list of node names visited
    budget: int           # max number of engineer invocations
    spent_budget: int     # how many times concierge has routed to data_engineer
    go_to_engineer: bool  # set by concierge to drive its conditional edge
    concierge_done: bool
    engineer_done: bool
    analyst_done: bool
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
    spent = state.get("spent_budget", 0)
    budget = state.get("budget", 1)
    go = spent < budget
    return {
        **state,
        "route": state.get("route", []) + ["concierge"],
        "concierge_done": True,
        "go_to_engineer": go,
        "spent_budget": spent + 1 if go else spent,
    }


def data_engineer_node(state: V4State) -> V4State:
    return {**state, "route": state.get("route", []) + ["data_engineer"], "engineer_done": True}


def data_analyst_node(state: V4State) -> V4State:
    return {
        **state,
        "route": state.get("route", []) + ["data_analyst"],
        "analyst_done": True,
        "output_text": f"Analysis pass {state.get('spent_budget', 0)} complete for: {state.get('question', '')}",
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_input_guardrail(state: V4State) -> str:
    return "concierge" if state.get("input_guardrail_passed") else END


def _route_concierge(state: V4State) -> str:
    return "data_engineer" if state.get("go_to_engineer") else "output_guardrail"


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(V4State)

    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("concierge", concierge_node)
    g.add_node("data_engineer", data_engineer_node)
    g.add_node("data_analyst", data_analyst_node)
    g.add_node("output_guardrail", output_guardrail_node)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", _route_input_guardrail, ["concierge", END])
    g.add_conditional_edges("concierge", _route_concierge, ["data_engineer", "output_guardrail"])
    g.add_edge("data_engineer", "data_analyst")
    g.add_edge("data_analyst", "concierge")
    g.add_edge("output_guardrail", END)

    return g.compile()


graph_v4 = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_analysis_v4(question: str, *_args, budget: int = 3, **_kwargs) -> dict:
    """Run the V4 budget-controlled pipeline. budget sets max engineer invocations."""
    initial: V4State = {
        "question": question,
        "output_text": "",
        "route": [],
        "budget": budget,
        "spent_budget": 0,
        "go_to_engineer": False,
        "concierge_done": False,
        "engineer_done": False,
        "analyst_done": False,
        "input_guardrail_passed": False,
        "input_guardrail_reason": "",
        "output_guardrail_passed": False,
        "output_guardrail_reason": "",
    }

    s = graph_v4.invoke(initial)

    skipped = "— skipped"
    inp_blocked = not s["input_guardrail_passed"]
    rows = [
        ("Input Guardrail", "✓ Pass" if s["input_guardrail_passed"] else f"✗ Blocked — {s['input_guardrail_reason']}"),
        ("Concierge",       skipped if inp_blocked else f"✓ ({s['spent_budget']}/{s['budget']} budget used)"),
        ("Data Engineer",   skipped if inp_blocked else ("✓" if s["engineer_done"] else "✗")),
        ("Data Analyst",    skipped if inp_blocked else ("✓" if s["analyst_done"] else "✗")),
        ("Output Guardrail", skipped if inp_blocked else ("✓ Pass" if s["output_guardrail_passed"] else f"✗ Blocked — {s['output_guardrail_reason']}")),
    ]
    table = "\n".join(f"| {agent} | {status} |" for agent, status in rows)

    route_lines = "\n".join(
        f"{i + 1}. `{node}`" for i, node in enumerate(s.get("route", []))
    )

    analysis = (
        "**V4 Agent Pipeline — Final State**\n\n"
        "**Route taken:**\n\n"
        f"{route_lines}\n\n"
        "| Agent | Status |\n"
        "|---|---|\n"
        f"{table}"
    )

    return {"analysis": analysis, "charts": {}, "follow_up": []}
