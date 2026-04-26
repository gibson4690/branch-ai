import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data import generate_data, compute_highlights, BRANCHES, METRIC_META

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Branch.ai",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] > .main { background: #f7f8fc; }
  [data-testid="stHeader"] { display: none !important; }
  [data-testid="block-container"] { padding-top: 13px !important; }

  div[data-testid="stHorizontalBlock"] { align-items: flex-start; }

  /* ── Custom chat form (replaces st.chat_input) ── */
  div[data-testid="stForm"] {
    background: #fff !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 16px !important;
    padding: 0.15rem 0.35rem 0.15rem 0.9rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.055) !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    margin-top: 0.6rem !important;
  }
  div[data-testid="stForm"]:focus-within {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.09) !important;
  }
  div[data-testid="stForm"] input[type="text"] {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    font-size: 0.9rem !important;
    padding: 0.45rem 0 !important;
    color: #111827 !important;
  }
  div[data-testid="stForm"] input[type="text"]::placeholder { color: #9ca3af !important; }
  div[data-testid="stForm"] label { display: none !important; }
  div[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 11px !important;
    height: 38px !important;
    width: 38px !important;
    min-height: 38px !important;
    padding: 0 !important;
    font-size: 1.05rem !important;
    line-height: 1 !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.22) !important;
    transition: opacity 0.15s, box-shadow 0.15s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
  }
  div[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.32) !important;
  }
  div[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button:focus {
    box-shadow: 0 0 0 3px rgba(79,70,229,0.2) !important;
  }
  /* Keep columns inside the form vertically centred */
  div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 0.3rem !important;
  }

  /* ── 3D insight cards ── */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 4px 12px rgba(0,0,0,0.10), 0 1px 3px rgba(0,0,0,0.08) !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    cursor: pointer !important;
    background: white !important;
    position: relative !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.14), 0 2px 6px rgba(0,0,0,0.10) !important;
  }
  /* Tight gap between stacked insight cards */
  div[data-testid="element-container"]:has(> div[data-testid="stVerticalBlockBorderWrapper"]) {
    margin-bottom: -0.6rem !important;
  }
  /* Uniform padding inside bordered card */
  div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
    padding: 0.6rem !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] .stButton {
    display: flex !important; align-items: center !important;
    justify-content: center !important; height: 100% !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] .stButton button {
    width: 30px !important; height: 30px !important; min-height: 30px !important;
    padding: 0 !important; opacity: 1 !important;
    background: #f3f4f6 !important; border: 1px solid #e5e7eb !important;
    border-radius: 8px !important; color: #9ca3af !important;
    font-size: 0.85rem !important; line-height: 1 !important;
    transition: background 0.15s, color 0.15s, border-color 0.15s !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] .stButton button:hover {
    background: #e0e7ff !important; color: #4f46e5 !important;
    border-color: #c7d2fe !important;
  }

  /* ── Suggested question pill buttons ── */
  .stButton button[kind="primary"] {
    background: #f8f9ff !important;
    border: 1.5px solid #c7d2fe !important;
    border-radius: 20px !important;
    color: #4338ca !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 1rem !important;
    transition: all 0.15s ease !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    text-align: center !important;
  }
  .stButton button[kind="primary"]:hover {
    background: #e0e7ff !important;
    border-color: #818cf8 !important;
    color: #3730a3 !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.15) !important;
    transform: translateY(-1px) !important;
  }

  /* ── Analysis content typography ── */
  .analysis-body ul, .analysis-body ol {
    margin-left: 1.4rem !important;
    margin-top: 0.3rem !important;
    margin-bottom: 0.5rem !important;
  }
  .analysis-body li { margin-bottom: 0.3rem !important; line-height: 1.7 !important; }
  .analysis-body p  { margin-bottom: 0.55rem !important; line-height: 1.65 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
df = generate_data()
positives, negatives = compute_highlights(df)


def _v5_datasets() -> dict:
    """Split the global df into the three thematic datasets used by the V5 agent."""
    perf_cols  = ["branch", "month", "avg_wait_time", "avg_handling_time", "counter_utilization"]
    queue_cols = ["branch", "month", "queue_tokens", "missed_queue", "total_transactions"]
    staff_cols = ["branch", "month", "staff_seedling", "staff_sapling", "staff_mature",
                  "senior_pct", "corporate_clients", "retail_customers"]
    return {
        "performance": df[perf_cols],
        "queue":       df[queue_cols],
        "staff":       df[staff_cols],
    }

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULT_QUESTIONS = [
    "Which branch has the longest wait times?",
    "How is counter utilisation trending?",
    "Which branches are losing the most customers?",
]

for key, default in [
    ("chat_messages", []),
    ("pending_analysis", None),
    ("_scroll_to_chat", False),
    ("suggested_questions", _DEFAULT_QUESTIONS),
    ("agent_mode", "Multi-Agent"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar: agent architecture toggle ───────────────────────────────────────
def _on_agent_mode_change():
    for k in [k for k in st.session_state if k.startswith("agent_cache_")]:
        del st.session_state[k]



# ── Chart helper ──────────────────────────────────────────────────────────────
def _generate_chart(spec: dict, highlight_branch: str = None):
    chart_type  = spec.get("type", "line")
    metric      = spec.get("metric")
    metric_y    = spec.get("metric_y")
    branches    = spec.get("branches") or []
    title       = spec.get("title", "")
    months_back = spec.get("months_back")

    if not metric or metric not in df.columns:
        return None

    meta  = METRIC_META.get(metric, {})
    label = meta.get("label", metric)
    unit  = meta.get("unit", "")
    asc   = meta.get("lower_is_better", False)

    # Optionally restrict to a recent time window
    if months_back:
        latest_m = df["month"].max()
        cutoff   = latest_m - pd.DateOffset(months=int(months_back) - 1)
        base = df[df["month"] >= cutoff]
    else:
        base = df

    subset = base[base["branch"].isin(branches)] if branches else base

    _layout = dict(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=36, b=0),
    )

    if chart_type == "line":
        fig = px.line(
            subset.sort_values("month"), x="month", y=metric, color="branch",
            title=title or f"{label} — Monthly Trend",
            labels={"month": "", metric: f"{label} ({unit})"},
        )
        if highlight_branch:
            for trace in fig.data:
                trace.opacity = 1.0 if trace.name == highlight_branch else 0.25
                if trace.name == highlight_branch:
                    trace.line.width = 3
        fig.update_layout(
            height=300,
            legend=dict(orientation="h", y=-0.28, font_size=11),
            **_layout,
        )
        return fig

    if chart_type == "area":
        fig = px.area(
            subset.sort_values("month"), x="month", y=metric, color="branch",
            title=title or f"{label} — Trend",
            labels={"month": "", metric: f"{label} ({unit})"},
        )
        if highlight_branch:
            for trace in fig.data:
                trace.opacity = 1.0 if trace.name == highlight_branch else 0.18
        fig.update_layout(
            height=300,
            legend=dict(orientation="h", y=-0.28, font_size=11),
            **_layout,
        )
        return fig

    if chart_type == "bar":
        snapshot = base[base["month"] == base["month"].max()].copy()
        if branches:
            snapshot = snapshot[snapshot["branch"].isin(branches)]
        fig = px.bar(
            snapshot.sort_values(metric, ascending=asc),
            x="branch", y=metric,
            title=title or f"{label} — Branch Comparison",
            labels={"branch": "", metric: unit},
            color=metric,
            color_continuous_scale="RdYlGn_r" if asc else "RdYlGn",
        )
        fig.update_layout(
            height=280, coloraxis_showscale=False, xaxis_tickangle=-30,
            **_layout,
        )
        return fig

    if chart_type == "ranking":
        agg = subset.groupby("branch")[metric].mean().reset_index()
        fig = px.bar(
            agg.sort_values(metric, ascending=not asc),
            x=metric, y="branch", orientation="h",
            title=title or f"{label} — Branch Ranking",
            labels={"branch": "", metric: f"{label} ({unit})"},
            color=metric,
            color_continuous_scale="RdYlGn_r" if asc else "RdYlGn",
        )
        fig.update_layout(
            height=max(220, 36 * len(agg) + 60),
            coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
            **_layout,
        )
        return fig

    if chart_type == "scatter":
        if not metric_y or metric_y not in df.columns:
            return None
        meta_y  = METRIC_META.get(metric_y, {})
        label_y = meta_y.get("label", metric_y)
        unit_y  = meta_y.get("unit", "")
        agg = subset.groupby("branch")[[metric, metric_y]].mean().reset_index()
        fig = px.scatter(
            agg, x=metric, y=metric_y, text="branch",
            title=title or f"{label} vs {label_y}",
            labels={metric: f"{label} ({unit})", metric_y: f"{label_y} ({unit_y})"},
            color="branch",
        )
        fig.update_traces(textposition="top center", marker_size=10)
        fig.update_layout(
            height=320,
            legend=dict(orientation="h", y=-0.28, font_size=11),
            **_layout,
        )
        return fig

    if chart_type == "heatmap":
        pivot = (
            subset.groupby(["branch", subset["month"].dt.strftime("%b %Y")])[metric]
            .mean()
            .unstack()
        )
        # Keep chronological column order
        all_months = (
            subset["month"].drop_duplicates()
            .sort_values()
            .dt.strftime("%b %Y")
            .tolist()
        )
        cols_ordered = [c for c in all_months if c in pivot.columns]
        pivot = pivot[cols_ordered]

        colorscale = "RdYlGn_r" if asc else "RdYlGn"
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=colorscale,
            hovertemplate="%{y} · %{x}<br>" + f"{label}: %{{z:.1f}} {unit}<extra></extra>",
        ))
        fig.update_layout(
            title=title or f"{label} — Branch × Time Heatmap",
            height=max(240, 36 * len(pivot) + 80),
            xaxis=dict(tickangle=-45, tickfont_size=10),
            **_layout,
        )
        return fig

    return None


# ── Inline analysis + chart renderer ─────────────────────────────────────────
def _render_inline_analysis(analysis: str, charts: dict, highlight_branch: str = None):
    """Render analysis text with [CHART:id] markers replaced by inline charts.
    Any charts not referenced by a marker are appended at the end."""
    if not charts:
        st.markdown(analysis)
        return

    parts = re.split(r'\[CHART:(\w+)\]', analysis)
    rendered_ids = set()

    # parts alternates: text_segment, chart_id, text_segment, chart_id, ...
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            spec = charts.get(part)
            rendered_ids.add(part)
            if spec:
                fig = _generate_chart(spec, highlight_branch=highlight_branch)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Append any charts the LLM didn't place a marker for
    leftover = [s for cid, s in charts.items() if cid not in rendered_ids]
    if leftover:
        st.divider()
        _WIDE_TYPES = {"heatmap", "scatter", "area"}
        grid = [s for s in leftover if s.get("type") not in _WIDE_TYPES]
        wide = [s for s in leftover if s.get("type") in _WIDE_TYPES]
        if grid:
            cols = st.columns(min(len(grid), 2))
            for i, spec in enumerate(grid):
                fig = _generate_chart(spec, highlight_branch=highlight_branch)
                if fig:
                    with cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        for spec in wide:
            fig = _generate_chart(spec, highlight_branch=highlight_branch)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Question pill renderer ────────────────────────────────────────────────────
def _render_question_pills(questions: list, key_prefix: str, label: str = "Suggested questions"):
    st.markdown(f"""
    <div style="font-size:0.7rem;font-weight:700;color:#9ca3af;letter-spacing:0.07em;
                text-transform:uppercase;margin:0.8rem 0 0.4rem;">
      {label}
    </div>""", unsafe_allow_html=True)
    cols = st.columns(len(questions))
    for i, q in enumerate(questions):
        with cols[i]:
            if st.button(q, key=f"{key_prefix}_{i}", use_container_width=True, type="primary"):
                st.session_state.chat_messages.append({"role": "user", "content": q})
                st.session_state.pending_analysis = {"question": q}
                st.session_state._scroll_to_chat = True
                st.rerun()


# ── Deep dive renderer ────────────────────────────────────────────────────────
def _render_deep_dive(ctx: dict):
    question = ctx.get("question", "Performance Analysis")
    branch   = ctx.get("branch", "")
    metric   = ctx.get("metric", "")
    label    = METRIC_META[metric]["label"] if metric else ""

    if branch and metric:
        st.markdown(f"#### {label} — {branch} Branch")
        st.caption(f"Question: {question}")
    else:
        st.markdown(f"#### {question}")

    mode = st.session_state.get("agent_mode", "ReAct Agent")
    cache_key = f"agent_cache_{mode}_{abs(hash(str(ctx)))}"

    if cache_key not in st.session_state:
        try:
            from llm import _get_api_key

            if _get_api_key():
                placeholder = st.empty()

                if mode == "Multi-Agent":
                    placeholder.markdown(
                        "_Running multi-agent pipeline: DataConcierge → DataEngineer → DataAnalyst → Executive…_ ▌"
                    )
                    from agents_v2 import run_analysis_v2
                    result = run_analysis_v2(question, df, ctx)
                elif mode == "V3 Agent":
                    placeholder.markdown(
                        "_Running V3 pipeline: Concierge → DataAnalyst → DataEngineer → DataAnalyst…_ ▌"
                    )
                    from agents_v3 import run_analysis_v3
                    result = run_analysis_v3(question, df, ctx)
                elif mode == "V4 Agent":
                    placeholder.markdown(
                        "_Running V4 pipeline: Concierge → DataAnalyst → DataEngineer → DataAnalyst…_ ▌"
                    )
                    from agents_v4 import run_analysis_v4
                    result = run_analysis_v4(question, df, ctx)
                elif mode == "V5 Agent":
                    placeholder.markdown(
                        "_Running V5 pipeline: InputGuardrail → PrepareState → Concierge → DataEngineer → DataAnalyst → Reviewer…_ ▌"
                    )
                    from agents_v5 import run_analysis_v5
                    result = run_analysis_v5(question, df, ctx)
                else:
                    placeholder.markdown("_Analysing data…_ ▌")
                    from agents import run_analysis
                    result = run_analysis(question, df, ctx)

                placeholder.empty()
                st.session_state[cache_key] = result
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": result.get("analysis", ""),
                    "charts": result.get("charts", {}),
                    "branch": branch,
                    "insight_plots": result.get("insight_plots", []),
                })
                if result.get("follow_up"):
                    st.session_state.suggested_questions = result["follow_up"]
            else:
                result = {"analysis": "_AI analysis requires `ANTHROPIC_API_KEY`._", "charts": {}, "follow_up": []}
                st.session_state[cache_key] = result

        except Exception as e:
            result = {"analysis": f"_Analysis error: {e}_", "charts": {}, "follow_up": []}
            st.session_state[cache_key] = result

    result    = st.session_state[cache_key]
    analysis  = result.get("analysis", str(result)) if isinstance(result, dict) else str(result)
    charts    = result.get("charts", {})    if isinstance(result, dict) else {}
    follow_up = result.get("follow_up", []) if isinstance(result, dict) else []

    if mode == "V5 Agent":
        st.markdown(analysis)
        insight_plots = result.get("insight_plots", []) if isinstance(result, dict) else []
        if insight_plots:
            from agents_v5 import generate_plot_from_instruction
            datasets = _v5_datasets()
            for plots in insight_plots:
                for instr in plots:
                    fig = generate_plot_from_instruction(instr, datasets)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    elif charts:
        _render_inline_analysis(analysis, charts, highlight_branch=branch or None)
    elif metric:
        st.divider()
        fig = _generate_chart(
            {"type": "line", "metric": metric, "branches": [], "title": f"{label} — All Branches"},
            highlight_branch=branch,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        fig2 = _generate_chart({"type": "bar", "metric": metric, "branches": [],
                                 "title": f"{label} — Branch Comparison ({df['month'].max().strftime('%b %Y')})"})
        if fig2:
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.divider()
        net = df.groupby("month")[["avg_wait_time", "missed_queue", "total_transactions"]].mean().reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(net, x="month", y="avg_wait_time", title="Network Avg Wait Time",
                          labels={"month": "", "avg_wait_time": "Minutes"})
            fig.update_layout(height=240, margin=dict(l=0, r=0, t=36, b=0),
                              plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            fig2 = px.line(net, x="month", y="missed_queue", title="Network Missed Queue Count",
                           labels={"month": "", "missed_queue": "Customers"})
            fig2.update_layout(height=240, margin=dict(l=0, r=0, t=36, b=0),
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})



# ── Sparkline helper ──────────────────────────────────────────────────────────
def _sparkline(values: list, line_color: str, fill_color: str):
    fig = go.Figure(go.Scatter(
        x=list(range(len(values))), y=values,
        mode="lines", line=dict(color=line_color, width=2),
        fill="tozeroy", fillcolor=fill_color,
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=2, b=0), height=52,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Banner (fixed) ────────────────────────────────────────────────────────────
st.markdown(f"""
<div id="branch-banner" style="
    position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
    background: rgba(247,248,252,0.97);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(229,231,235,0.8);
    box-shadow: 0 1px 10px rgba(0,0,0,0.06);
    padding: 0.6rem 3rem;
    display: flex; align-items: center; gap: 1rem;">
  <div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);border-radius:11px;
              padding:0.52rem;display:flex;align-items:center;justify-content:center;
              box-shadow:0 3px 12px rgba(79,70,229,0.30);">
    <svg width="26" height="26" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="15" cy="6"  r="2.8" fill="white"/>
      <circle cx="6"  cy="22" r="2.8" fill="white"/>
      <circle cx="24" cy="22" r="2.8" fill="white"/>
      <line x1="15" y1="8.8"  x2="8"  y2="19.2" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="15" y1="8.8"  x2="22" y2="19.2" stroke="white" stroke-width="1.8" stroke-linecap="round"/>
      <line x1="6"  y1="24.8" x2="24" y2="24.8" stroke="white" stroke-width="1.4"
            stroke-linecap="round" opacity="0.35"/>
    </svg>
  </div>
  <div style="flex:1;">
    <div style="font-size:1.65rem;font-weight:900;letter-spacing:-0.04em;line-height:1;
                font-family:'Inter','SF Pro Display','Segoe UI',sans-serif;">
      <span style="color:#111827;">Branch</span><span
        style="background:linear-gradient(135deg,#4f46e5,#7c3aed);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;">.ai</span>
    </div>
    <div style="font-size:0.78rem;color:#6b7280;margin-top:0.18rem;">
      Welcome, Country Manager &nbsp;·&nbsp;
      {pd.Timestamp.now().strftime('%A, %d %B %Y')} &nbsp;·&nbsp;
      {len(BRANCHES)} branches &nbsp;·&nbsp;
      Data: Jan 2024 – {df['month'].max().strftime('%b %Y')}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Highlights — 60 % page width ─────────────────────────────────────────────
_, highlights_col, _ = st.columns([2, 6, 2])
with highlights_col:
    col_pos, col_neg = st.columns(2, gap="large")


def _render_cards(insights: list, is_positive: bool, col):
    lc     = "#16a34a" if is_positive else "#dc2626"
    fill   = "rgba(22,163,74,0.07)" if is_positive else "rgba(220,38,38,0.07)"
    hdr_bg = "#f0fdf4" if is_positive else "#fef2f2"
    bdr    = "#d1fae5" if is_positive else "#fee2e2"
    title  = "Top Positive Highlights" if is_positive else "Top Negative Alerts"
    arrow  = "↑" if is_positive else "↓"

    with col:
        st.markdown(f"""
        <div style="background:{hdr_bg};border:1px solid {bdr};border-radius:8px;
                    padding:0.1rem;font-size:0.8rem;font-weight:700;color:{lc};
                    letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.5rem;">
            {arrow}&nbsp; {title}
        </div>""", unsafe_allow_html=True)

        for ins in insights:
            pct = ins["pct_change"]
            lib = ins["lower_is_better"]
            badge = f"{'↓' if pct < 0 else '↑'} {abs(pct):.1f}%"
            verb  = ("decreased" if pct < 0 else "increased") if lib else ("increased" if pct > 0 else "decreased")

            with st.container(border=True):
                c_text, c_chart, c_btn = st.columns([4, 2, 1], vertical_alignment="top")
                with c_text:
                    st.markdown(f"""<div style="display:flex;align-items:center;padding:0.1rem;margin:0.1rem;">
                        <span style="font-weight:600;font-size:0.85rem;color:#111;">{ins['label']}</span>
                        <span style="font-size:0.78rem;color:#888;margin-left:0.5rem;">{ins['branch']}</span>
                        </div>""", unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='font-size:0.8rem;color:#444;line-height:1.5;margin:0.1rem;'>"
                        f"{ins['branch']} <b>{ins['label'].lower()}</b> has {verb} "
                        f"<b>{abs(pct):.1f}%</b>, now at <b>{ins['current']:.1f} {ins['unit']}</b>.</p>",
                        unsafe_allow_html=True,
                    )
                with c_chart:
                    st.markdown(
                        f"<div style='text-align:center;margin-bottom:0.1rem;'>"
                        f"<span style='background:{lc};color:#fff;border-radius:10px;"
                        f"padding:1px 8px;font-size:0.72rem;font-weight:700;'>{badge}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    branch_vals = (
                        df[df["branch"] == ins["branch"]]
                        .sort_values("month")[ins["metric"]]
                        .tolist()[-9:]
                    )
                    st.plotly_chart(
                        _sparkline(branch_vals, lc, fill),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                with c_btn:
                    if st.button("→", key=f"btn_{ins['branch']}_{ins['metric']}"):
                        st.session_state.pending_analysis = {
                            "question": f"Analyse {ins['label']} performance at {ins['branch']} branch",
                            "branch":   ins["branch"],
                            "metric":   ins["metric"],
                        }
                        st.session_state._scroll_to_chat = True
                        st.rerun()



_render_cards(positives, True,  col_pos)
_render_cards(negatives, False, col_neg)

# ── JS: wire whole-card click to hidden trigger button ────────────────────────
st.markdown("""
<script>
(function() {
  function wireCards() {
    document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]').forEach(function(card) {
      if (card._branchWired) return;
      card._branchWired = true;
      card.addEventListener('click', function(e) {
        if (e.target.closest('iframe')) return;
        var btn = card.querySelector('.stButton button');
        if (btn) btn.click();
      });
    });
  }
  wireCards();
  new MutationObserver(wireCards).observe(document.body, {childList: true, subtree: true});
})();
</script>""", unsafe_allow_html=True)

# ── Agent selector ────────────────────────────────────────────────────────────
_, agent_sel_col, _ = st.columns([2, 6, 2])
with agent_sel_col:
    st.selectbox(
        "Analysis engine:",
        options=["ReAct Agent", "Multi-Agent", "V3 Agent", "V4 Agent", "V5 Agent"],
        key="agent_mode",
        on_change=_on_agent_mode_change,
        help="Switch between agent architectures.",
    )

# ── Chat box — 60 % page width ────────────────────────────────────────────────
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

_, chat_col, _ = st.columns([2, 6, 2])

# ── Helpers for custom message bubbles ───────────────────────────────────────
_USER_BUBBLE = """
<div style="display:flex;justify-content:flex-end;margin:0.55rem 0 0.2rem;">
  <div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);color:#fff;
              border-radius:18px 18px 4px 18px;padding:0.7rem 1.05rem;
              max-width:78%;font-size:0.875rem;line-height:1.55;
              box-shadow:0 2px 10px rgba(79,70,229,0.18);">
    {content}
  </div>
</div>"""

_AI_HEADER = """
<div style="display:flex;align-items:center;gap:0.45rem;margin:0.9rem 0 0.25rem;">
  <div style="width:27px;height:27px;border-radius:8px;flex-shrink:0;
              background:linear-gradient(135deg,#4f46e5,#7c3aed);
              display:flex;align-items:center;justify-content:center;font-size:13px;">🏦</div>
  <span style="font-size:0.74rem;font-weight:700;color:#6b7280;letter-spacing:0.04em;
               text-transform:uppercase;">Branch.ai</span>
  {badge}
</div>"""

_AI_BUBBLE = """
<div style="margin-left:36px;background:#fff;border:1px solid #eaecf0;
            border-radius:4px 16px 16px 16px;padding:0.8rem 1.05rem;
            box-shadow:0 1px 4px rgba(0,0,0,0.05);font-size:0.875rem;
            line-height:1.6;color:#1f2937;margin-bottom:0.3rem;">
  {content}
</div>"""

def _ai_header(badge_label: str = "", badge_color: str = "#7c3aed") -> str:
    badge = (
        f'<span style="background:{badge_color};color:#fff;border-radius:8px;'
        f'padding:1px 7px;font-size:0.68rem;font-weight:600;">{badge_label}</span>'
        if badge_label else ""
    )
    return _AI_HEADER.format(badge=badge)

def _render_analysis_block(content: str, charts: dict, branch: str = ""):
    """Avatar header + analysis with inline charts (no HTML wrapper — Streamlit renders charts)."""
    mode = st.session_state.get("agent_mode", "Multi-Agent")
    badge = "Multi-Agent" if mode == "Multi-Agent" else "ReAct"
    badge_color = "#7c3aed" if mode == "Multi-Agent" else "#4f46e5"
    st.markdown(_ai_header(badge, badge_color), unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="analysis-body">', unsafe_allow_html=True)
        _render_inline_analysis(content, charts, highlight_branch=branch or None)
        st.markdown('</div>', unsafe_allow_html=True)


with chat_col:
    _active_mode = st.session_state.get("agent_mode", "Multi-Agent")
    _mode_badge_color = "#7c3aed" if _active_mode == "Multi-Agent" else "#4f46e5"
    _mode_label = "Multi-Agent" if _active_mode == "Multi-Agent" else "ReAct"

    # ── Section header ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div id="chat-section" style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.9rem;">
      <div style="width:34px;height:34px;border-radius:10px;flex-shrink:0;
                  background:linear-gradient(135deg,#4f46e5,#7c3aed);
                  display:flex;align-items:center;justify-content:center;">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
             fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <div>
        <div style="font-size:0.95rem;font-weight:800;color:#111827;letter-spacing:-0.02em;line-height:1.15;">
          Ask the Data
        </div>
        <div style="font-size:0.71rem;color:#9ca3af;margin-top:0.05rem;">powered by Branch.ai</div>
      </div>
      <span style="margin-left:auto;background:{_mode_badge_color};color:#fff;
                   border-radius:10px;padding:2px 10px;font-size:0.71rem;font-weight:700;
                   letter-spacing:0.04em;text-transform:uppercase;">{_mode_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Message history ───────────────────────────────────────────────────────
    for msg in st.session_state.chat_messages[-6:]:
        if msg["role"] == "user":
            st.markdown(
                _USER_BUBBLE.format(content=msg["content"]),
                unsafe_allow_html=True,
            )
        else:
            content       = msg.get("content", "")
            charts        = msg.get("charts", {})
            branch        = msg.get("branch", "")
            insight_plots = msg.get("insight_plots", [])
            if charts:
                _render_analysis_block(content, charts, branch)
            else:
                st.markdown(_ai_header(), unsafe_allow_html=True)
                st.markdown(
                    _AI_BUBBLE.format(content=content),
                    unsafe_allow_html=True,
                )
            if insight_plots:
                from agents_v5 import generate_plot_from_instruction
                datasets = _v5_datasets()
                for plots in insight_plots:
                    for instr in plots:
                        fig = generate_plot_from_instruction(instr, datasets)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Deep dive — rendered HERE, before input & suggestions ─────────────────
    if st.session_state.pending_analysis:
        ctx = st.session_state.pending_analysis
        mode = st.session_state.get("agent_mode", "Multi-Agent")
        badge = "Multi-Agent" if mode == "Multi-Agent" else "ReAct"
        badge_color = "#7c3aed" if mode == "Multi-Agent" else "#4f46e5"
        st.markdown(_ai_header(badge, badge_color), unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="analysis-body">', unsafe_allow_html=True)
            _render_deep_dive(ctx)
            st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.pending_analysis = None
        st.rerun()

    # ── Custom input bar ──────────────────────────────────────────────────────
    with st.form("chat_input_form", clear_on_submit=True):
        col_txt, col_btn = st.columns([11, 1])
        with col_txt:
            user_input = st.text_input(
                "chat",
                placeholder="Ask about branch performance — e.g. 'Which branches have the worst wait times?'",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("↑", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})
        st.session_state.pending_analysis = {"question": user_input.strip()}
        st.session_state._scroll_to_chat = True
        st.rerun()

    # ── Suggested questions ───────────────────────────────────────────────────
    _sq_label = "Quick questions" if not st.session_state.chat_messages else "Suggested follow-ups"
    _sq_key   = f"sq_{abs(hash(tuple(st.session_state.suggested_questions)))}"
    _render_question_pills(st.session_state.suggested_questions, key_prefix=_sq_key, label=_sq_label)

# Scroll to chat section after card click or question submission
if st.session_state.get("_scroll_to_chat"):
    st.markdown("""
    <script>
      setTimeout(function() {
        var el = document.getElementById('chat-section');
        if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
      }, 120);
    </script>""", unsafe_allow_html=True)
    st.session_state._scroll_to_chat = False
