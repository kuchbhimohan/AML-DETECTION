"""Sentinel AML — transaction-monitoring dashboard.

Run:  uv run streamlit run app/streamlit_app.py
Data: app/data/ (built by `uv run python -m aml.demo_data`); every number comes from reports/results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from ui import C

st.set_page_config(page_title="Sentinel AML · Transaction Monitoring", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

DATA = Path(__file__).parent / "data"
CHAMP = "LightGBM monotone (champion)"
LOGREG = "Logistic regression"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data
def load():
    j = lambda f: json.loads((DATA / f).read_text())
    alerts = pd.read_parquet(DATA / "alerts.parquet")
    return alerts, j("results.json"), j("curves.json"), j("profile.json"), j("extra_models.json"), \
        j("explainability.json"), j("ablation_age.json"), j("model_meta.json"), j("calibration_bins.json")


alerts, R, CURVES, PROFILE, EXTRA, EXPL, ABL, META, CAL_BINS = load()
N_TX, N_POS, DAYS = R["meta"]["test_rows"], R["meta"]["test_laundering"], CURVES["days"]
TX_DAY, POS_DAY = N_TX / DAYS, N_POS / DAYS


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


NICE = {"payment_method": "Payment format (ACH etc.)", "pair_history": "History between the two accounts",
        "log_amount_usd": "Amount", "payment_currency": "Currency", "s_age_hours": "Sender first seen (hours)",
        "r_age_hours": "Receiver first seen (hours)", "s_in_usd_24h": "Money sender received, 24h",
        "s_in_cnt_24h": "Payments sender received, 24h", "s_in_uniq_24h": "Distinct payers of sender, 24h",
        "s_hours_since_prev_out": "Hours since sender's last payment", "s_out_cnt_1h": "Sender payments, 1h",
        "s_out_cnt_24h": "Sender payments, 24h", "s_out_cnt_7d": "Sender payments, 7d", "s_out_uniq_1h": "Distinct payees, 1h",
        "s_out_uniq_24h": "Distinct payees, 24h", "s_out_uniq_7d": "Distinct payees, 7d", "s_out_usd_24h": "Sender sent, 24h",
        "s_out_usd_7d": "Sender sent, 7d", "s_amt_z_7d": "Amount vs usual (z)", "s_amt_ratio_7d": "Amount vs usual (×)",
        "r_in_cnt_1h": "Receiver payments in, 1h", "r_in_cnt_24h": "Receiver payments in, 24h", "r_in_cnt_7d": "Receiver payments in, 7d",
        "r_in_uniq_1h": "Receiver distinct payers, 1h", "r_in_uniq_24h": "Receiver distinct payers, 24h",
        "r_in_uniq_7d": "Receiver distinct payers, 7d", "r_in_usd_24h": "Receiver received, 24h", "r_out_cnt_24h": "Receiver onward payments, 24h",
        "r_out_uniq_24h": "Receiver distinct payees, 24h", "s_passthrough_ratio_24h": "Pass-through ratio, 24h",
        "reverse_flow_7d": "Money flowed back (7d)", "round_amount": "Round amount", "cross_bank": "Cross-bank",
        "self_transfer": "Self-transfer", "cross_currency": "Cross-currency", "hour": "Hour of day"}
LOG_TICKS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def log_axis(fig, title: str):
    """Readable log-scale percentage axis (explicit ticks instead of crowded auto-ticks)."""
    fig.update_xaxes(type="log", title=title, tickvals=LOG_TICKS,
                     ticktext=[f"{100 * t:g}%" for t in LOG_TICKS], range=[np.log10(1e-4), np.log10(0.1)])


def interp(curve_name: str, metric: str, budget: float) -> float:
    return float(np.interp(np.log(budget), np.log(CURVES["budget"]), CURVES[curve_name][metric]))


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
PAGES = ["◎  Command Center", "⚑  Alert Queue", "⌕  Investigate", "◧  Budget Simulator", "⚗  Model Lab"]
with st.sidebar:
    ui.brand()
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]
    page = st.radio("Navigation", PAGES, key="page")
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    show_truth = st.toggle("Reveal investigation outcome", value=False,
                           help="Shows the simulator's ground-truth label. Analysts would not see this.")
    st.markdown(f"""<div class="panel" style="margin-top:1rem; font-size:.8rem; line-height:1.55">
      <div class="kpi-label" style="margin-bottom:.4rem">Model card</div>
      <b>Monotone LightGBM</b> · 38 leak-free features<br>
      Train Sep 1–6 · Tune Sep 7–8 · <b>Test Sep 9–10</b><br>
      {N_TX:,} test transfers · {N_POS:,} laundering<br>
      Calibrated (isotonic) · SHAP reason codes</div>""", unsafe_allow_html=True)
    st.markdown("<div class='note' style='margin-top:.8rem'>Data: IBM synthetic AML transactions (HI-Small). "
                "All metrics from the untouched test period.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 1. Command center
# --------------------------------------------------------------------------- #
def command_center():
    ci = R["bootstrap_ci"][CHAMP]["recall_at_1pct"]
    d = R["bootstrap_diff"][f"{CHAMP} - {LOGREG}"]["recall_at_1pct"]
    f1 = R["f1_operating_point"]
    ui.hero("Anti-money-laundering · test period",
            f'Catches <span class="hl">{pct(ci[0])}</span> of laundering while reviewing just <span class="hl">1%</span> of transfers',
            f"Leak-free behavioural features, a monotone gradient-boosted model and calibrated, explainable alerts — "
            f"evaluated on {N_TX:,} unseen bank transfers with bootstrap confidence intervals and significance tests.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.kpi("Laundering caught @1%", pct(ci[0]), f"95% CI <b>{pct(ci[1])}–{pct(ci[2])}</b>", C["cyan"])
    with c2:
        ui.kpi("Lift vs logistic", f"+{100 * d[0]:.1f} pts",
               f"CI +{100 * d[1]:.1f} to +{100 * d[2]:.1f} · McNemar p={R['mcnemar_1pct_on_laundering']['p_value']:.0e}", C["violet"])
    with c3:
        ui.kpi("High-risk precision", pct(f1["test_precision"]), f"<b>{f1['alerts']}</b> high-risk alerts · recall {pct(f1['test_recall'])}", C["red"])
    with c4:
        tm = R["test_metrics"]
        ui.kpi("PR-AUC", f"{tm[CHAMP]['pr_auc']:.3f}", f"logistic <b>{tm[LOGREG]['pr_auc']:.3f}</b> · rules <b>{tm['Rules (ACH, amount)']['pr_auc']:.3f}</b>", C["amber"])

    left, right = st.columns([1.55, 1])
    with left:
        ui.section("Laundering caught vs. share of transfers reviewed", "Cumulative gains on the full test period (log scale).")
        fig = go.Figure()
        for (name, col) in zip(["LightGBM (champion)", "Logistic regression", "Rules (ACH, amount)"], ui.SERIES):
            fig.add_trace(go.Scatter(x=CURVES["budget"], y=CURVES[name]["recall"], name=name, mode="lines",
                                     line=dict(width=3 if "champion" in name else 2, color=col),
                                     hovertemplate="review %{x:.2%} → catch %{y:.1%}<extra>" + name + "</extra>"))
        fig.add_vline(x=0.01, line=dict(color=C["muted"], dash="dot"), annotation_text="1% budget",
                      annotation_font_color=C["muted"])
        log_axis(fig, "share of transfers reviewed")
        fig.update_yaxes(tickformat=".0%", title="share of laundering caught", range=[0, 1.02])
        st.plotly_chart(ui.style(fig, 380), width="stretch", config={"displayModeBar": False})
    with right:
        ui.section("Recall by laundering pattern", "At the 1% alert budget, with Wilson 95% CIs.")
        pr = R["pattern_recall_1pct"]
        names = sorted(pr, key=lambda k: pr[k]["champion"])
        fig = go.Figure()
        fig.add_trace(go.Bar(y=names, x=[pr[n]["champion"] for n in names], orientation="h", name="Champion",
                             marker=dict(color=[C["cyan"] if pr[n]["champion"] >= 0.9 else C["amber"] if pr[n]["champion"] >= 0.6 else C["red"]
                                                for n in names]),
                             error_x=dict(type="data", symmetric=False, color=C["muted"], thickness=1,
                                          array=[pr[n]["champion_ci"][1] - pr[n]["champion"] for n in names],
                                          arrayminus=[pr[n]["champion"] - pr[n]["champion_ci"][0] for n in names]),
                             customdata=[pr[n]["n"] for n in names],
                             hovertemplate="%{y}: %{x:.1%} (n=%{customdata})<extra></extra>"))
        fig.add_trace(go.Scatter(y=names, x=[pr[n]["logreg"] for n in names], mode="markers", name="Logistic regression",
                                 marker=dict(symbol="line-ns-open", size=16, color="white", line=dict(width=2.5)),
                                 hovertemplate="logistic %{x:.1%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(ui.style(fig, 380), width="stretch", config={"displayModeBar": False})
    st.markdown("<div class='note'>Known patterns are caught 96–100% of the time; the hardest group is laundering the "
                "simulator did not tag with a pattern (37.5%) — the model's honest weak spot.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 2. Alert queue
# --------------------------------------------------------------------------- #
def alert_queue():
    ui.hero("Investigator workspace", "Alert queue",
            f"The {len(alerts):,} highest-risk transfers of the test period, ranked by calibrated probability. "
            "Select a row, then open <b>Investigate</b> to see why it was flagged.")
    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.2, 1.6])
    bands = f1.multiselect("Risk band", ["High", "Medium"], default=["High", "Medium"])
    fmts = f2.multiselect("Payment format", sorted(alerts["payment_format"].unique()))
    min_p = f3.slider("Min. probability", 0.0, 1.0, 0.0, 0.01, format="%.2f")
    q = f4.text_input("Search reasons", placeholder="e.g. fan-out, ACH, first-ever")
    df = alerts[alerts["band"].isin(bands) & (alerts["p_cal"] >= min_p)]
    if fmts:
        df = df[df["payment_format"].isin(fmts)]
    if q:
        m = df[["reason_1", "reason_2", "reason_3"]].apply(lambda s: s.str.contains(q, case=False, na=False)).any(axis=1)
        df = df[m]

    k1, k2, k3, k4 = st.columns(4)
    with k1: ui.kpi("Alerts shown", f"{len(df):,}", f"of {len(alerts):,} in queue", C["violet"])
    with k2: ui.kpi("High risk", f"{int((df['band'] == 'High').sum()):,}", "best-F1 threshold", C["red"])
    with k3: ui.kpi("Median amount", f"${df['amount_usd'].median():,.0f}" if len(df) else "—", "USD equivalent", C["amber"])
    with k4:
        if show_truth and len(df):
            ui.kpi("Confirmed laundering", pct(df["label"].mean()), f"<b>{int(df['label'].sum())}</b> of {len(df):,} (ground truth)", C["cyan"])
        else:
            ui.kpi("ACH share", pct((df["payment_format"] == "ACH").mean()) if len(df) else "—", "of shown alerts", C["cyan"])

    view = pd.DataFrame({"Rank": df["rank"], "Risk": df["p_cal"], "Band": df["band"],
                         "Amount (USD)": df["amount_usd"], "Format": df["payment_format"], "Currency": df["payment_currency"],
                         "Time": pd.to_datetime(df["timestamp"]).dt.strftime("%b %d %H:%M"), "Top reason": df["reason_1"],
                         "2nd reason": df["reason_2"]})
    if show_truth:
        view["Outcome"] = np.where(df["label"] == 1, "Laundering · " + df["pattern"].fillna(""), "Legitimate")
    styler = view.style.map(lambda b: f"color:{ui.BAND_COLOR.get(b, C['muted'])}; font-weight:700", subset=["Band"])
    if show_truth:
        styler = styler.map(lambda o: f"color:{C['red'] if o.startswith('Laundering') else C['green']}", subset=["Outcome"])
    event = st.dataframe(styler, hide_index=True, height=520, width="stretch", on_select="rerun", selection_mode="single-row",
                         column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="percent"),
                                        "Amount (USD)": st.column_config.NumberColumn(format="dollar"),
                                        "Rank": st.column_config.NumberColumn(width="small")})
    sel = event.selection.rows if event and event.selection else []
    if sel:
        st.session_state["selected_rank"] = int(view.iloc[sel[0]]["Rank"])
        c1, c2 = st.columns([3, 1])
        c1.success(f"Alert #{st.session_state['selected_rank']} selected.")
        c2.button("Investigate →", type="primary", width="stretch",
                  on_click=lambda: st.session_state.update(page=PAGES[2]))


# --------------------------------------------------------------------------- #
# 3. Investigate
# --------------------------------------------------------------------------- #
@st.cache_resource
def explainer():
    from aml.explain import GROUPS, prepare, shap_values
    return GROUPS, prepare, shap_values


def investigate():
    ranks = alerts["rank"].tolist()
    default = st.session_state.get("selected_rank", ranks[0])
    rank = st.selectbox("Alert", ranks, index=ranks.index(default) if default in ranks else 0,
                        format_func=lambda r: f"#{r} · {alerts.loc[alerts['rank'] == r, 'band'].iat[0]} risk · "
                                              f"${alerts.loc[alerts['rank'] == r, 'amount_usd'].iat[0]:,.0f} "
                                              f"{alerts.loc[alerts['rank'] == r, 'payment_format'].iat[0]}")
    st.session_state["selected_rank"] = rank
    row = alerts[alerts["rank"] == rank].iloc[0]
    reasons = json.loads(row["reasons_json"])

    st.markdown(f"""<div class="hero" style="padding:1.3rem 1.6rem"><div class="eyebrow">Alert #{rank} · test period</div>
      <h1 style="font-size:1.6rem">${row['amount_usd']:,.0f} via {row['payment_format']} &nbsp; {ui.chip(row['band'])}</h1>
      <p>{pd.to_datetime(row['timestamp']):%A %d %b %Y, %H:%M} · {row['payment_currency']} · ranked {rank:,} of {N_TX:,} transfers</p></div>""",
                unsafe_allow_html=True)

    g, r, f = st.columns([1, 1.45, 1.05])
    with g:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * row["p_cal"],
                                     number=dict(suffix="%", font=dict(family="JetBrains Mono", size=40)),
                                     gauge=dict(axis=dict(range=[0, 100], tickcolor=C["muted"]), bar=dict(color=ui.BAND_COLOR[row["band"]], thickness=.28),
                                                bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                                steps=[dict(range=[0, 100 * 0.007], color="rgba(61,220,151,.18)"),
                                                       dict(range=[100 * 0.007, 30], color="rgba(255,181,71,.16)"),
                                                       dict(range=[30, 100], color="rgba(255,77,109,.16)")])))
        st.plotly_chart(ui.style(fig, 250, legend=False), width="stretch", config={"displayModeBar": False})
        st.markdown("<div class='note' style='text-align:center'>Calibrated P(laundering)</div>", unsafe_allow_html=True)
    with r:
        ui.section("Why it was flagged", "Top reasons that pushed the risk up (SHAP).")
        ui.reason_list(reasons)
    with f:
        ui.section("Transaction facts")
        ui.facts([("Sender payments, last 1h", f"{row['s_out_cnt_1h']:.0f}"),
                  ("Distinct payees, last 24h", f"{row['s_out_uniq_24h']:.0f}"),
                  ("Receiver's distinct payers, 7d", f"{row['r_in_uniq_7d']:.0f}"),
                  ("Earlier payments in this pair", f"{row['pair_prev_cnt']:.0f}"),
                  ("Sender received, last 24h", f"{row['s_in_cnt_24h']:.0f} payments"),
                  ("Cross-bank", "yes" if row["cross_bank"] else "no")])

    c1, c2 = st.columns([1.25, 1])
    with c1:
        ui.section("Full explanation", "Every feature's push on the score, in log-odds (red raises risk, green lowers it).")
        GROUPS, prepare, shap_values = explainer()
        contrib, base = shap_values(prepare(alerts[alerts["rank"] == rank]))
        s = contrib.iloc[0]
        for g_name, members in GROUPS.items():
            s[g_name] = s[members].sum(); s = s.drop(members)
        top = s.reindex(s.abs().sort_values(ascending=False).index[:10])[::-1]
        fig = go.Figure(go.Bar(x=top.values, y=[NICE.get(i, i) for i in top.index], orientation="h",
                               marker_color=[C["red"] if v > 0 else C["green"] for v in top.values],
                               hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        fig.update_xaxes(title="SHAP contribution (log-odds)")
        st.plotly_chart(ui.style(fig, 380, legend=False), width="stretch", config={"displayModeBar": False})
    with c2:
        ui.section("How unusual is this behaviour?", "Percentile of this transfer among normal traffic.")
        labels = {"s_out_uniq_24h": "Distinct payees 24h", "s_out_cnt_1h": "Payments last hour", "s_out_cnt_24h": "Payments 24h",
                  "r_in_uniq_24h": "Receiver payers 24h", "r_in_uniq_7d": "Receiver payers 7d", "s_in_cnt_24h": "Money received 24h",
                  "log_amount_usd": "Amount", "s_hours_since_prev_out": "Hours since last payment",
                  "pair_prev_cnt": "Pair history", "s_amt_ratio_7d": "Amount vs usual"}
        pcts = {labels[f]: float(np.searchsorted(PROFILE[f], row[f], side="right") - 1) for f in PROFILE}
        items = sorted(pcts.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                               marker=dict(color=[v for _, v in items], colorscale=[[0, C["blue"]], [.5, C["violet"]], [1, C["red"]]], cmin=0, cmax=100),
                               hovertemplate="%{y}: %{x:.0f}th percentile<extra></extra>"))
        fig.add_vline(x=50, line=dict(color=C["muted"], dash="dot"))
        fig.update_xaxes(range=[0, 100], title="percentile vs normal transfers")
        st.plotly_chart(ui.style(fig, 380, legend=False), width="stretch", config={"displayModeBar": False})

    if show_truth:
        is_l = row["label"] == 1
        col = C["red"] if is_l else C["green"]
        txt = f"Confirmed laundering · pattern: {row['pattern']}" if is_l else "Legitimate transfer (false alarm)"
        st.markdown(f"<div class='truth' style='background:{col}18; border:1px solid {col}66; color:{col}'>Ground truth: {txt}</div>",
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 4. Budget simulator
# --------------------------------------------------------------------------- #
def simulator():
    ui.hero("Operations planning", "Alert-budget simulator",
            "How many transfers should investigators review? Trade review cost against missed laundering. "
            "Curves come from the full test period.")
    a, b, c, d = st.columns(4)
    budget = a.select_slider("Alert budget (share of transfers reviewed)",
                             options=[0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05],
                             value=0.01, format_func=lambda x: f"{100 * x:g}%")
    review_cost = b.number_input("Cost of one review ($)", 1, 1000, 25, 5)
    miss_cost = c.number_input("Cost of a missed case ($)", 100, 1_000_000, 10_000, 1000)
    capacity = d.number_input("Reviews per analyst per day", 10, 500, 60, 10)

    rec = interp("LightGBM (champion)", "recall", budget); prec = interp("LightGBM (champion)", "precision", budget)
    rec_l = interp("Logistic regression", "recall", budget)
    alerts_day = budget * TX_DAY; caught_day = rec * POS_DAY
    k1, k2, k3, k4 = st.columns(4)
    with k1: ui.kpi("Laundering caught", pct(rec), f"logistic <b>{pct(rec_l)}</b> at same budget", C["cyan"])
    with k2: ui.kpi("Alerts per day", f"{alerts_day:,.0f}", f"precision <b>{pct(prec)}</b>", C["violet"])
    with k3: ui.kpi("Analysts needed", f"{np.ceil(alerts_day / capacity):,.0f}", f"at {capacity} reviews/day", C["amber"])
    cost = alerts_day * review_cost + (POS_DAY - caught_day) * miss_cost
    with k4: ui.kpi("Daily cost", f"${cost / 1000:,.0f}k", f"vs <b>${POS_DAY * miss_cost / 1000:,.0f}k</b> with no monitoring", C["red"])

    bs = np.array(CURVES["budget"])
    costs = {n: bs * TX_DAY * review_cost + (1 - np.array(CURVES[n]["recall"])) * POS_DAY * miss_cost
             for n in ["LightGBM (champion)", "Logistic regression"]}
    best_i = int(np.argmin(costs["LightGBM (champion)"]))
    l, r = st.columns(2)
    with l:
        ui.section("Total daily cost vs. alert budget", f"Cost-optimal budget for the champion: <b>{pct(bs[best_i], 2)}</b> of transfers.")
        fig = go.Figure()
        for (n, col) in zip(costs, ui.SERIES):
            fig.add_trace(go.Scatter(x=bs, y=costs[n] / 1000, name=n, line=dict(color=col, width=3 if "champion" in n else 2)))
        fig.add_trace(go.Scatter(x=[bs[best_i]], y=[costs["LightGBM (champion)"][best_i] / 1000], mode="markers", name="optimum",
                                 marker=dict(size=13, color=C["cyan"], line=dict(color="white", width=2))))
        fig.add_vline(x=budget, line=dict(color=C["muted"], dash="dot"))
        log_axis(fig, "alert budget"); fig.update_yaxes(title="daily cost ($k)")
        st.plotly_chart(ui.style(fig, 360), width="stretch", config={"displayModeBar": False})
    with r:
        ui.section("Precision–recall trade-off", "Each point is an alert budget; the dot is your current choice.")
        fig = go.Figure()
        for (n, col) in zip(["LightGBM (champion)", "Logistic regression"], ui.SERIES):
            fig.add_trace(go.Scatter(x=CURVES[n]["recall"], y=CURVES[n]["precision"], name=n, line=dict(color=col, width=2.5)))
        fig.add_trace(go.Scatter(x=[rec], y=[prec], mode="markers", name="current budget",
                                 marker=dict(size=13, color=C["amber"], line=dict(color="white", width=2))))
        fig.update_xaxes(tickformat=".0%", title="recall"); fig.update_yaxes(tickformat=".0%", title="precision", type="log")
        st.plotly_chart(ui.style(fig, 360), width="stretch", config={"displayModeBar": False})
    st.markdown(f"<div class='note'>Per day: {TX_DAY:,.0f} transfers, {POS_DAY:,.0f} laundering (test-period averages). "
                "Costs are illustrative inputs; the curves are measured.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 5. Model lab
# --------------------------------------------------------------------------- #
def model_lab():
    ui.hero("Evidence", "Model lab", "How the champion compares, whether the differences are real, and whether the "
            "probabilities can be trusted. Everything below is on the untouched test period.")
    t1, t2, t3, t4 = st.tabs(["Model comparison", "Calibration", "Drivers (SHAP)", "Drift & robustness"])
    with t1:
        m = EXTRA["metrics"]; diff = EXTRA["diff_vs_champion"]
        rows = []
        for name, x in sorted(m.items(), key=lambda kv: -kv[1]["recall_at_1pct"]):
            dd = diff.get(f"{name} - {CHAMP}", {}).get("recall_at_1pct")
            sig = "champion" if dd is None else ("tie" if dd[1] <= 0 <= dd[2] else ("better" if dd[1] > 0 else "worse"))
            rows.append({"Model": name, "PR-AUC": x["pr_auc"], "Recall @1%": x["recall_at_1pct"],
                         "Δ vs champion": np.nan if dd is None else dd[0], "95% CI": "—" if dd is None else f"[{dd[1]:+.3f}, {dd[2]:+.3f}]",
                         "Verdict": sig})
        lr = R["test_metrics"][LOGREG]; dl = R["bootstrap_diff"][f"{CHAMP} - {LOGREG}"]["recall_at_1pct"]
        rows.append({"Model": LOGREG, "PR-AUC": lr["pr_auc"], "Recall @1%": lr["recall_at_1pct"], "Δ vs champion": -dl[0],
                     "95% CI": f"[{-dl[2]:+.3f}, {-dl[1]:+.3f}]", "Verdict": "worse"})
        tab = pd.DataFrame(rows)
        col = {"better": C["cyan"], "tie": C["muted"], "worse": C["red"], "champion": C["violet"]}
        st.dataframe(tab.style.map(lambda v: f"color:{col.get(v, C['text'])}; font-weight:700", subset=["Verdict"]),
                     hide_index=True, width="stretch",
                     column_config={"PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                                    "Recall @1%": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                                    "Δ vs champion": st.column_config.NumberColumn(format="%+.3f")})
        dlg = R["delong"]["champion_vs_logreg"]
        st.markdown(f"<div class='note'>Paired Poisson bootstrap (CI including 0 = tie). Champion vs logistic: DeLong z = {dlg['z']:.2f}, "
                    f"p = {dlg['p_value']:.1e}; McNemar p = {R['mcnemar_1pct_on_laundering']['p_value']:.1e}. "
                    "Alternative models use fixed settings (no tuning on test); the champion was chosen on validation.</div>",
                    unsafe_allow_html=True)
    with t2:
        cm = R["calibration"]["methods"]
        l, r = st.columns([1.2, 1])
        with l:
            fig = go.Figure()
            names = list(cm)
            fig.add_trace(go.Bar(x=names, y=[cm[n]["brier"] * 1e4 for n in names], name="Brier ×10⁴", marker_color=C["violet"]))
            fig.add_trace(go.Bar(x=names, y=[cm[n]["ece"] * 1e4 for n in names], name="ECE ×10⁴", marker_color=C["cyan"]))
            fig.update_layout(barmode="group"); fig.update_yaxes(title="lower is better")
            st.plotly_chart(ui.style(fig, 340), width="stretch", config={"displayModeBar": False})
        with r:
            ui.facts([(n, f"mean {cm[n]['mean_pred'] * 100:.3f}%") for n in names] +
                     [("Actual laundering rate", f"{R['calibration']['actual_rate'] * 100:.3f}%")])
            st.markdown(f"<div class='note' style='margin-top:.6rem'>Isotonic calibration (fitted on validation) gives the best Brier "
                        f"score and cuts ECE by a third. Platt's slope of {R['calibration']['platt_params']['slope_on_logit']:.2f} "
                        "shows the prior-corrected model was already close to calibrated. A constant predictor has the "
                        "lowest ECE but zero ranking power — calibration metrics need ranking metrics beside them.</div>",
                        unsafe_allow_html=True)
        ui.section("Does “70% risk” really mean ~70%?", "Calibrated probability bins vs. what actually happened, "
                   "top-2,500 alerts of the test period.")
        cb = pd.DataFrame(CAL_BINS)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cb["bin"], y=cb["actual"], name="actually laundering", marker_color=C["cyan"],
                             text=[f"{v:.0%}<br>n={n}" for v, n in zip(cb["actual"], cb["n"])], textposition="outside"))
        fig.add_trace(go.Scatter(x=cb["bin"], y=cb["mean_predicted"], name="mean predicted", mode="markers",
                                 marker=dict(symbol="diamond", size=12, color=C["amber"], line=dict(color="white", width=1))))
        fig.update_yaxes(tickformat=".0%", range=[0, 1.18])
        st.plotly_chart(ui.style(fig, 340), width="stretch", config={"displayModeBar": False})
        st.markdown("<div class='note'>Probabilities are capped at 99%: the top isotonic step was 127/127 laundering on "
                    "test, but no model should claim certainty.</div>", unsafe_allow_html=True)
    with t3:
        imp = pd.Series(EXPL["importance_all"]).sort_values(ascending=False).head(15)[::-1]
        direc = EXPL["direction"]
        fig = go.Figure(go.Bar(x=imp.values, y=[NICE.get(f, f) for f in imp.index], orientation="h",
                               marker_color=[C["red"] if direc.get(f, 0) > 0.1 else C["green"] if direc.get(f, 0) < -0.1 else C["violet"] for f in imp.index],
                               hovertemplate="%{y}: mean |SHAP| %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="mean |SHAP| across all test traffic (log-odds)")
        st.plotly_chart(ui.style(fig, 460, legend=False), width="stretch", config={"displayModeBar": False})
        st.markdown(f"<div class='note'>Red: higher value raises risk · green: lowers it · violet: mixed/categorical. "
                    f"All 7 monotone constraints show the expected direction; {EXPL['logreg_agreement']['agree']} of "
                    f"{EXPL['logreg_agreement']['n']} significant logistic coefficients agree in sign.</div>", unsafe_allow_html=True)
    with t4:
        dr = R["drift"]
        l, r = st.columns([1.3, 1])
        with l:
            p = pd.Series(dr["feature_psi_top10"])[::-1]
            fig = go.Figure(go.Bar(x=p.values, y=p.index, orientation="h",
                                   marker_color=[C["red"] if v > .25 else C["amber"] if v > .1 else C["green"] for v in p.values]))
            fig.add_vline(x=0.1, line=dict(color=C["amber"], dash="dot")); fig.add_vline(x=0.25, line=dict(color=C["red"], dash="dot"))
            fig.update_xaxes(type="log", title="PSI train → test (log)")
            st.plotly_chart(ui.style(fig, 380, legend=False), width="stretch", config={"displayModeBar": False})
        with r:
            ui.facts([("Score PSI, validation → test", f"{dr['score_psi_val_to_test']:.3f}"),
                      ("Score PSI, train → test", f"{dr['score_psi_train_to_test']:.3f}"),
                      ("Base rate train → test", f"{dr['base_rate_train'] * 100:.3f}% → {dr['base_rate_test'] * 100:.3f}%"),
                      ("Val PR-AUC with age features", f"{ABL['with']['pr_auc']:.3f}"),
                      ("Val PR-AUC without age features", f"{ABL['without']['pr_auc']:.3f}"),
                      ("Val recall@1% with / without", f"{ABL['with']['recall_at_1pct'] * 100:.1f}% / {ABL['without']['recall_at_1pct'] * 100:.1f}%")])
            st.markdown("<div class='note' style='margin-top:.6rem'>The data begins on Sep 1, so account-age and long-window "
                        "features keep growing in the first week (history warm-up). The score itself is stable from validation "
                        "to test, and removing the clock-like age features leaves the 1%-budget recall almost unchanged.</div>",
                        unsafe_allow_html=True)


{PAGES[0]: command_center, PAGES[1]: alert_queue, PAGES[2]: investigate, PAGES[3]: simulator, PAGES[4]: model_lab}[page]()
