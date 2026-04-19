import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from data import generate_data, compute_highlights, BRANCHES, METRIC_META

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Branch.ai",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] > .main { background: #f7f8fc; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stChatInput"] textarea { border-radius: 12px; }
  div[data-testid="stHorizontalBlock"] { align-items: flex-start; }

  /* 3D card effect on bordered containers */
  div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 4px 12px rgba(0,0,0,0.10), 0 1px 3px rgba(0,0,0,0.08) !important;
    border-radius: 12px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    cursor: pointer !important;
    background: white !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.14), 0 2px 6px rgba(0,0,0,0.10) !important;
  }

  /* Style "Analyse" trigger as minimal text link inside cards */
  div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"] {
    background: transparent !important;
    border: none !important;
    color: #6b7280 !important;
    font-size: 0.75rem !important;
    padding: 0 !important;
    box-shadow: none !important;
    text-decoration: underline !important;
  }
  div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"]:hover {
    color: #111827 !important;
    background: transparent !important;
  }

  /* Deep dive analysis section */
  .deep-dive-wrap {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.4rem 1.6rem 1rem;
    margin-top: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
</style>
""", unsafe_allow_html=True)

# ── Data ───────────────────────────────────────────────────────────────────────
df = generate_data()
positives, negatives = compute_highlights(df)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("chat_messages", []),
    ("pending_analysis", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Chart helper ───────────────────────────────────────────────────────────────
def _generate_chart(spec: dict, highlight_branch: str = None):
    chart_type = spec.get("type", "line")
    metric     = spec.get("metric")
    branches   = spec.get("branches")
    title      = spec.get("title", "")

    if not metric or metric not in df.columns:
        return None

    meta  = METRIC_META.get(metric, {})
    label = meta.get("label", metric)
    unit  = meta.get("unit", "")
    asc   = meta.get("lower_is_better", False)

    subset = df[df["branch"].isin(branches)] if branches else df

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
            height=300, margin=dict(l=0, r=0, t=36, b=0),
            legend=dict(orientation="h", y=-0.28, font_size=11),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        return fig

    if chart_type == "bar":
        latest = df[df["month"] == df["month"].max()].copy()
        if branches:
            latest = latest[latest["branch"].isin(branches)]
        fig = px.bar(
            latest.sort_values(metric, ascending=asc),
            x="branch", y=metric,
            title=title or f"{label} — Branch Comparison",
            labels={"branch": "", metric: unit},
            color=metric,
            color_continuous_scale="RdYlGn_r" if asc else "RdYlGn",
        )
        fig.update_layout(
            height=280, margin=dict(l=0, r=0, t=36, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False, xaxis_tickangle=-30,
        )
        return fig

    return None


# ── Inline deep dive renderer ──────────────────────────────────────────────────
def _render_deep_dive(ctx: dict):
    question = ctx.get("question", "Performance Analysis")
    branch   = ctx.get("branch", "")
    metric   = ctx.get("metric", "")
    label    = METRIC_META[metric]["label"] if metric else ""

    st.markdown('<div class="deep-dive-wrap">', unsafe_allow_html=True)

    if branch and metric:
        st.markdown(f"#### {label} — {branch} Branch")
        st.caption(f"Question: {question}")
    else:
        st.markdown(f"#### {question}")

    cache_key = f"llm_{abs(hash(str(ctx)))}"

    if cache_key not in st.session_state:
        # Stream the analysis text
        try:
            import anthropic
            from llm import _get_api_key
            import json, re, os
            from data import METRIC_META as MM

            api_key = _get_api_key()
            if api_key:
                latest_m = df["month"].max()
                recent   = df[df["month"] >= latest_m - pd.DateOffset(months=2)]
                summary  = recent.groupby("branch")[[
                    "avg_wait_time", "missed_queue", "total_transactions",
                    "counter_utilization", "corporate_clients", "senior_pct",
                    "staff_seedling", "staff_sapling", "staff_mature",
                ]].mean().round(1)

                focus = ""
                if branch and metric:
                    unit_str = MM.get(metric, {}).get("unit", "")
                    lbl      = MM.get(metric, {}).get("label", metric)
                    vals     = df[df["branch"] == branch].sort_values("month")[metric].tolist()
                    focus = (
                        f"\nFocus: {lbl} at {branch} branch.\n"
                        f"27-month history (Jan 2024–Mar 2026): {[round(v, 1) for v in vals]} {unit_str}\n"
                    )

                available = [k for k in MM if k not in ("staff_seedling", "staff_sapling", "staff_mature")]

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

                client = anthropic.Anthropic(api_key=api_key)

                # Stream and accumulate
                full_text = ""
                placeholder = st.empty()
                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=700,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text_chunk in stream.text_stream:
                        full_text += text_chunk
                        # Show raw accumulation while streaming
                        placeholder.markdown(full_text + "▌")

                placeholder.empty()

                # Parse JSON from streamed result
                m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", full_text)
                text_to_parse = m.group(1) if m else full_text
                try:
                    result = json.loads(text_to_parse)
                    result.setdefault("analysis", full_text)
                    result.setdefault("charts", [])
                except json.JSONDecodeError:
                    result = {"analysis": full_text, "charts": []}

                st.session_state[cache_key] = result
                if not branch and not metric:
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": result.get("analysis", "")}
                    )
            else:
                result = {"analysis": "_AI analysis requires `ANTHROPIC_API_KEY`._", "charts": []}
                st.session_state[cache_key] = result

        except Exception as e:
            result = {"analysis": f"_Analysis error: {e}_", "charts": []}
            st.session_state[cache_key] = result

    result = st.session_state[cache_key]
    analysis_text = result.get("analysis", str(result)) if isinstance(result, dict) else str(result)
    st.markdown(analysis_text)

    st.divider()

    charts = result.get("charts", []) if isinstance(result, dict) else []

    if charts:
        cols = st.columns(min(len(charts), 2))
        for i, spec in enumerate(charts[:4]):
            fig = _generate_chart(spec, highlight_branch=branch or None)
            if fig:
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    elif metric:
        fig = _generate_chart(
            {"type": "line", "metric": metric, "branches": None, "title": f"{label} — All Branches"},
            highlight_branch=branch,
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        fig2 = _generate_chart({"type": "bar", "metric": metric, "branches": None,
                                 "title": f"{label} — Branch Comparison ({df['month'].max().strftime('%b %Y')})"})
        if fig2:
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
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

    st.markdown('</div>', unsafe_allow_html=True)


# ── Sparkline helper ───────────────────────────────────────────────────────────
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


# ── Banner ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 1.6rem 0 0.6rem;">
  <div style="font-size: 1.8rem; font-weight: 800; color: #111827; letter-spacing: -0.03em;">
    Branch.ai
  </div>
  <div style="font-size: 0.88rem; color: #6b7280; margin-top: 0.2rem;">
    Welcome, General Manager &nbsp;·&nbsp;
    {pd.Timestamp.now().strftime('%A, %d %B %Y')} &nbsp;·&nbsp;
    {len(BRANCHES)} branches &nbsp;·&nbsp;
    Data: Jan 2024 – {df['month'].max().strftime('%b %Y')}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chat box ───────────────────────────────────────────────────────────────────
for msg in st.session_state.chat_messages[-4:]:
    avatar = "👔" if msg["role"] == "user" else "🏦"
    with st.chat_message(msg["role"], avatar=avatar):
        content = msg["content"]
        st.write(content if isinstance(content, str) else content)

prompt = st.chat_input(
    "Ask about branch performance — e.g. 'Which branches have the worst wait times?'"
)
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    st.session_state.pending_analysis = {"question": prompt}
    st.rerun()

# ── Inline deep dive (chat-originated) ────────────────────────────────────────
if st.session_state.pending_analysis:
    ctx = st.session_state.pending_analysis
    avatar = "🏦"
    with st.chat_message("assistant", avatar=avatar):
        _render_deep_dive(ctx)
    st.session_state.pending_analysis = None

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ── Highlights — 60% page width, two equal columns ────────────────────────────
_, highlights_col, _ = st.columns([2, 6, 2])

with highlights_col:
    col_pos, col_neg = st.columns(2, gap="large")


def _render_cards(insights: list, is_positive: bool, col):
    lc     = "#16a34a" if is_positive else "#dc2626"
    fill   = "rgba(22,163,74,0.07)" if is_positive else "rgba(220,38,38,0.07)"
    hdr_bg = "#f0fdf4" if is_positive else "#fef2f2"
    bdr    = "#d1fae5" if is_positive else "#fee2e2"
    title  = "Top Positive Highlights" if is_positive else "Top Negative Highlights"
    arrow  = "↑" if is_positive else "↓"

    with col:
        st.markdown(f"""
        <div style="background:{hdr_bg};border:1px solid {bdr};border-radius:8px;
                    padding:0.5rem 1rem;font-size:0.8rem;font-weight:700;color:{lc};
                    letter-spacing:0.05em;text-transform:uppercase;margin-bottom:0.9rem;">
            {arrow}&nbsp; {title}
        </div>""", unsafe_allow_html=True)

        for ins in insights:
            pct = ins["pct_change"]
            lib = ins["lower_is_better"]
            arrow_badge = "↓" if pct < 0 else "↑"
            badge = f"{arrow_badge} {abs(pct):.1f}%"

            if lib:
                verb = "decreased" if pct < 0 else "increased"
            else:
                verb = "increased" if pct > 0 else "decreased"

            with st.container(border=True):
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                    <div>
                        <span style="font-weight:600;font-size:0.85rem;color:#111;">
                            {ins['label']}
                        </span>
                        <span style="font-size:0.78rem;color:#888;margin-left:0.5rem;">
                            {ins['branch']}
                        </span>
                    </div>
                    <span style="background:{lc};color:#fff;border-radius:12px;
                                 padding:1px 9px;font-size:0.75rem;font-weight:600;">
                        {badge}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                c_text, c_chart = st.columns([4, 2])

                with c_text:
                    st.markdown(
                        f"<p style='font-size:0.8rem;color:#444;line-height:1.5;margin:0 0 0.3rem;'>"
                        f"{ins['branch']} <b>{ins['label'].lower()}</b> has {verb} "
                        f"<b>{abs(pct):.1f}%</b>, now at "
                        f"<b>{ins['current']:.1f} {ins['unit']}</b>.</p>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Analyse →", key=f"btn_{ins['branch']}_{ins['metric']}"):
                        st.session_state.pending_analysis = {
                            "question": f"Analyse {ins['label']} performance at {ins['branch']} branch",
                            "branch":   ins["branch"],
                            "metric":   ins["metric"],
                        }
                        st.rerun()

                with c_chart:
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

            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)


_render_cards(positives, True,  col_pos)
_render_cards(negatives, False, col_neg)
