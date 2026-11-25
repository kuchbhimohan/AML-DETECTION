"""Visual layer for the dashboard: CSS, reusable HTML components, Plotly theme."""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

C = {"bg": "#0A0F1E", "panel": "#111830", "panel2": "#161F3D", "line": "#24305A", "text": "#E7EAF3",
     "muted": "#8E98B8", "violet": "#7C6CFF", "cyan": "#2EE6D6", "red": "#FF4D6D", "amber": "#FFB547",
     "green": "#3DDC97", "blue": "#4DA3FF"}
BAND_COLOR = {"High": C["red"], "Medium": C["amber"], "Low": C["green"]}
SERIES = [C["violet"], C["cyan"], C["amber"], C["blue"], C["red"], C["green"]]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{ font-family: 'Inter', sans-serif !important; }}
#MainMenu, footer, header [data-testid="stToolbar"] {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 600px at 85% -10%, rgba(124,108,255,.16), transparent 60%),
              radial-gradient(900px 500px at -10% 10%, rgba(46,230,214,.08), transparent 60%), {C['bg']};
}}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0D1430 0%, #0A0F1E 100%); border-right: 1px solid {C['line']}; }}
[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .55rem .8rem; border-radius: 10px; margin-bottom: .15rem; width: 100%;
  border: 1px solid transparent; transition: all .15s ease; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(124,108,255,.10); border-color: {C['line']}; }}
[data-testid="stSidebar"] [role="radiogroup"] label[data-selected="true"],
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: linear-gradient(90deg, rgba(124,108,255,.28), rgba(124,108,255,.06)); border-color: rgba(124,108,255,.55); }}
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-weight: 600; font-size: .93rem; }}

.brand {{ display:flex; align-items:center; gap:.7rem; margin: .2rem 0 1.2rem 0; }}
.brand-logo {{ width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-weight:800;
  background: conic-gradient(from 210deg, {C['violet']}, {C['cyan']}, {C['violet']}); color:#0A0F1E; font-size:1.05rem;
  box-shadow: 0 0 24px rgba(124,108,255,.45); }}
.brand-name {{ font-weight:800; font-size:1.12rem; letter-spacing:.2px; }}
.brand-sub {{ color:{C['muted']}; font-size:.72rem; letter-spacing:1.6px; text-transform:uppercase; }}

.hero {{ position:relative; overflow:hidden; border-radius:20px; padding: 1.7rem 1.9rem; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, rgba(124,108,255,.22) 0%, rgba(17,24,48,.9) 45%, rgba(46,230,214,.10) 100%);
  border: 1px solid rgba(124,108,255,.35); }}
.hero:after {{ content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
  background: radial-gradient(circle, rgba(46,230,214,.25), transparent 70%); }}
.eyebrow {{ color:{C['cyan']}; font-size:.74rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; }}
.hero h1 {{ font-size:2.05rem; line-height:1.18; margin:.35rem 0 .45rem 0; font-weight:800; }}
.hero h1 .hl {{ background: linear-gradient(90deg, {C['cyan']}, {C['violet']}); -webkit-background-clip:text; color:transparent; }}
.hero p {{ color:{C['muted']}; margin:0; max-width: 880px; font-size:.98rem; }}

.kpi {{ background: linear-gradient(180deg, {C['panel2']}, {C['panel']}); border:1px solid {C['line']};
  border-radius:16px; padding:1rem 1.1rem 1.05rem 1.1rem; position:relative; overflow:hidden; height:100%; }}
.kpi:before {{ content:""; position:absolute; left:0; top:0; height:3px; width:100%; background: var(--accent); }}
.kpi-label {{ color:{C['muted']}; font-size:.74rem; font-weight:600; letter-spacing:1.2px; text-transform:uppercase; }}
.kpi-value {{ font-family:'JetBrains Mono', monospace; font-size:1.85rem; font-weight:700; margin:.25rem 0 .1rem 0; }}
.kpi-sub {{ color:{C['muted']}; font-size:.8rem; }}
.kpi-sub b {{ color:{C['text']}; font-weight:600; }}

.section {{ margin: 1.4rem 0 .6rem 0; }}
.section h3 {{ margin:0; font-size:1.12rem; font-weight:700; }}
.section p {{ margin:.15rem 0 0 0; color:{C['muted']}; font-size:.86rem; }}

.panel {{ background:{C['panel']}; border:1px solid {C['line']}; border-radius:16px; padding:1rem 1.15rem; }}
.chip {{ display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .62rem; border-radius:999px;
  font-size:.74rem; font-weight:700; letter-spacing:.4px; border:1px solid; }}
.chip:before {{ content:""; width:7px; height:7px; border-radius:50%; background: currentColor; }}
.reason {{ display:flex; gap:.8rem; align-items:flex-start; padding:.75rem .9rem; border-radius:12px; margin-bottom:.5rem;
  background: rgba(255,255,255,.025); border:1px solid {C['line']}; }}
.reason-n {{ min-width:28px; height:28px; border-radius:8px; display:grid; place-items:center; font-weight:800;
  background: rgba(255,77,109,.14); color:{C['red']}; font-family:'JetBrains Mono', monospace; }}
.reason-t {{ font-weight:600; font-size:.93rem; }}
.reason-s {{ color:{C['muted']}; font-size:.78rem; font-family:'JetBrains Mono', monospace; }}
.fact {{ display:flex; justify-content:space-between; padding:.45rem 0; border-bottom:1px dashed {C['line']}; font-size:.88rem; }}
.fact span:first-child {{ color:{C['muted']}; }}
.fact span:last-child {{ font-family:'JetBrains Mono', monospace; font-weight:600; }}
.truth {{ border-radius:12px; padding:.7rem .9rem; font-size:.88rem; font-weight:600; }}
.note {{ color:{C['muted']}; font-size:.8rem; }}
[data-testid="stDataFrame"] {{ border:1px solid {C['line']}; border-radius:14px; overflow:hidden; }}
div[data-baseweb="select"] > div, .stSlider {{ border-radius:10px !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
.stTabs [data-baseweb="tab"] {{ background:{C['panel']}; border:1px solid {C['line']}; border-radius:10px; padding:.35rem .9rem; }}
.stTabs [aria-selected="true"] {{ border-color: rgba(124,108,255,.6); background: rgba(124,108,255,.16); }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.markdown("""<div class="brand"><div class="brand-logo">S</div>
      <div><div class="brand-name">Sentinel AML</div><div class="brand-sub">Transaction monitoring</div></div></div>""",
                unsafe_allow_html=True)


def hero(eyebrow: str, title_html: str, sub: str):
    st.markdown(f"""<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div>
      <h1>{title_html}</h1><p>{sub}</p></div>""", unsafe_allow_html=True)


def kpi(label: str, value: str, sub_html: str = "", accent: str = C["violet"]):
    st.markdown(f"""<div class="kpi" style="--accent:{accent}"><div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value">{value}</div><div class="kpi-sub">{sub_html}</div></div>""", unsafe_allow_html=True)


def section(title: str, sub: str = ""):
    st.markdown(f"""<div class="section"><h3>{html.escape(title)}</h3>{f'<p>{sub}</p>' if sub else ''}</div>""",
                unsafe_allow_html=True)


def chip(band: str) -> str:
    col = BAND_COLOR.get(band, C["muted"])
    return f'<span class="chip" style="color:{col}; border-color:{col}55; background:{col}18">{html.escape(band)} risk</span>'


def reason_list(reasons: list[dict]):
    rows = "".join(f"""<div class="reason"><div class="reason-n">{i + 1}</div><div>
        <div class="reason-t">{html.escape(r['reason'])}</div>
        <div class="reason-s">SHAP {r['contribution']:+.2f} log-odds</div></div></div>""" for i, r in enumerate(reasons))
    st.markdown(rows, unsafe_allow_html=True)


def facts(items: list[tuple[str, str]]):
    st.markdown('<div class="panel">' + "".join(
        f'<div class="fact"><span>{html.escape(k)}</span><span>{html.escape(v)}</span></div>' for k, v in items)
        + "</div>", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=C["text"], size=12),
                      margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["line"], font=dict(family="Inter")))
    fig.update_xaxes(gridcolor="rgba(142,152,184,.12)", zerolinecolor="rgba(142,152,184,.2)", linecolor=C["line"])
    fig.update_yaxes(gridcolor="rgba(142,152,184,.12)", zerolinecolor="rgba(142,152,184,.2)", linecolor=C["line"])
    return fig
