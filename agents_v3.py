"""V3 agent graph with input/output guardrails.

Graph:
  START → input_guardrail ─(pass)→ concierge → data_analyst_1
                          ─(fail)→ END          → data_engineer → data_analyst_2
                                                → output_guardrail → END
"""
from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from llm import _get_api_key


# ── LLM helper ────────────────────────────────────────────────────────────────

def _llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=_get_api_key(), max_tokens=256)


# ── State ─────────────────────────────────────────────────────────────────────

class V3State(TypedDict):
    question: str
    output_text: str
    concierge_done: bool
    analyst_1_done: bool
    engineer_done: bool
    analyst_2_done: bool
    input_guardrail_passed: bool
    input_guardrail_reason: str
    output_guardrail_passed: bool
    output_guardrail_reason: str


# ── Guardrail nodes ───────────────────────────────────────────────────────────

def input_guardrail_node(state: V3State) -> V3State:
    question = state.get("question", "")
    prompt = (
        "You are a strict input guardrail for a bank branch analytics assistant.\n"
        "Evaluate the user question below and respond with exactly one line:\n"
        "  PASS – if the question is relevant to bank branch operations, performance, "
        "metrics, staff, queues, transactions, or customer insights.\n"
        "  BLOCK: <reason> – if the question is off-topic, harmful, violent, hateful, "
        "sexual, or otherwise inappropriate.\n\n"
        f"Question: {question}"
    )
    response = _llm().invoke(prompt).content.strip()
    passed = response.upper().startswith("PASS")
    reason = "" if passed else response.partition(":")[2].strip()
    return {**state, "input_guardrail_passed": passed, "input_guardrail_reason": reason}


def output_guardrail_node(state: V3State) -> V3State:
    text = state.get("output_text", "")
    prompt = (
        "You are a strict output guardrail for a bank branch analytics assistant.\n"
        "Evaluate the text below for personally identifiable information (PII) such as "
        "full names tied to data, NRIC/passport numbers, phone numbers, email addresses, "
        "home addresses, or any other sensitive personal details.\n"
        "Respond with exactly one line:\n"
        "  PASS – if no PII is present.\n"
        "  BLOCK: <reason> – if PII is detected, briefly describe what was found.\n\n"
        f"Text: {text}"
    )
    response = _llm().invoke(prompt).content.strip()
    passed = response.upper().startswith("PASS")
    reason = "" if passed else response.partition(":")[2].strip()
    return {**state, "output_guardrail_passed": passed, "output_guardrail_reason": reason}


# ── Agent nodes ───────────────────────────────────────────────────────────────

def concierge_node(state: V3State) -> V3State:
    return {**state, "concierge_done": True}


def data_analyst_node(state: V3State) -> V3State:
    if not state.get("analyst_1_done"):
        return {**state, "analyst_1_done": True}
    return {
        **state,
        "analyst_2_done": True,
        "output_text": f"Analysis complete for question: {state.get('question', '')}",
    }


def data_engineer_node(state: V3State) -> V3State:
    return {**state, "engineer_done": True}


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_input_guardrail(state: V3State) -> str:
    return "concierge" if state.get("input_guardrail_passed") else END


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(V3State)

    g.add_node("input_guardrail", input_guardrail_node)
    g.add_node("concierge", concierge_node)
    g.add_node("data_analyst_1", data_analyst_node)
    g.add_node("data_engineer", data_engineer_node)
    g.add_node("data_analyst_2", data_analyst_node)
    g.add_node("output_guardrail", output_guardrail_node)

    g.add_edge(START, "input_guardrail")
    g.add_conditional_edges("input_guardrail", _route_input_guardrail, ["concierge", END])
    g.add_edge("concierge", "data_analyst_1")
    g.add_edge("data_analyst_1", "data_engineer")
    g.add_edge("data_engineer", "data_analyst_2")
    g.add_edge("data_analyst_2", "output_guardrail")
    g.add_edge("output_guardrail", END)

    return g.compile()


graph_v3 = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def run_analysis_v3(question: str, *_args, **_kwargs) -> dict:
    """Run the V3 pipeline and return final state as formatted analysis."""
    initial: V3State = {
        "question": question,
        "output_text": "",
        "concierge_done": False,
        "analyst_1_done": False,
        "engineer_done": False,
        "analyst_2_done": False,
        "input_guardrail_passed": False,
        "input_guardrail_reason": "",
        "output_guardrail_passed": False,
        "output_guardrail_reason": "",
    }

    s = graph_v3.invoke(initial)

    if not s["input_guardrail_passed"]:
        analysis = (
            "**Input Guardrail — Blocked**\n\n"
            f"> {s['input_guardrail_reason']}\n\n"
            "Please ask a question related to bank branch operations and performance."
        )
        return {"analysis": analysis, "charts": {}, "follow_up": []}

    rows = [
        ("Input Guardrail",       "✓ Pass" if s["input_guardrail_passed"] else "✗ Blocked"),
        ("Concierge",             "✓" if s["concierge_done"] else "✗"),
        ("Data Analyst (pass 1)", "✓" if s["analyst_1_done"] else "✗"),
        ("Data Engineer",         "✓" if s["engineer_done"] else "✗"),
        ("Data Analyst (pass 2)", "✓" if s["analyst_2_done"] else "✗"),
        ("Output Guardrail",      "✓ Pass" if s["output_guardrail_passed"] else f"✗ Blocked — {s['output_guardrail_reason']}"),
    ]
    table = "\n".join(f"| {agent} | {status} |" for agent, status in rows)

    analysis = (
        "**V3 Agent Pipeline — Final State**\n\n"
        "| Agent | Status |\n"
        "|---|---|\n"
        f"{table}"
    )

    return {"analysis": analysis, "charts": {}, "follow_up": []}
