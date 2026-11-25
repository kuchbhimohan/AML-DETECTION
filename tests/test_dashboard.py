"""Headless tests of the Streamlit dashboard (streamlit.testing AppTest).

Runs the real app script, drives every page and widget, and fails on any
exception. Also checks that headline numbers shown match reports/results.json.
"""

import json

import pytest

from aml.config import PROJECT_ROOT

APP = PROJECT_ROOT / "app" / "streamlit_app.py"
DATA = PROJECT_ROOT / "app" / "data"
pytestmark = pytest.mark.skipif(not (DATA / "alerts.parquet").exists(), reason="dashboard data not built")

PAGES = ["◎  Command Center", "⚑  Alert Queue", "⌕  Investigate", "◧  Budget Simulator", "⚗  Model Lab"]


def _app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _goto(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, f"{page}: {at.exception}"
    return at


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


@pytest.fixture(scope="module")
def results():
    return json.loads((DATA / "results.json").read_text())


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("truth", [False, True])
def test_every_page_renders_without_errors(page, truth):
    at = _app()
    at.sidebar.toggle[0].set_value(truth).run()
    _goto(at, page)


def test_headline_numbers_match_results(results):
    at = _app()
    t = _text(at)
    champ = "LightGBM monotone (champion)"
    rec = results["bootstrap_ci"][champ]["recall_at_1pct"]
    assert f"{100 * rec[0]:.1f}%" in t and f"{100 * rec[1]:.1f}%–{100 * rec[2]:.1f}%" in t
    assert f"{100 * results['f1_operating_point']['test_precision']:.1f}%" in t
    assert f"{results['test_metrics'][champ]['pr_auc']:.3f}" in t
    assert f"{results['meta']['test_rows']:,}" in t


def test_alert_queue_empty_filter_does_not_crash():
    at = _goto(_app(), PAGES[1])
    at.slider[0].set_value(1.0).run()                     # nothing reaches p = 1.0 (capped at 0.99)
    assert not at.exception
    assert "of 2,500 in queue" in _text(at)
    assert "—" in _text(at)                                # median amount shows a dash, not NaN


def test_alert_queue_filters_reduce_rows():
    at = _goto(_app(), PAGES[1])
    at.multiselect[0].set_value(["High"]).run()           # risk band
    assert not at.exception
    t = _text(at)
    assert "572" in t                                      # 572 high-risk alerts (best-F1 operating point)


def test_investigate_any_alert_and_ground_truth():
    at = _app()
    at.sidebar.toggle[0].set_value(True).run()
    at = _goto(at, PAGES[2])
    for rank in (1, 434, 2500):
        at.selectbox[0].set_value(rank).run()
        assert not at.exception, rank
        t = _text(at)
        assert f"Alert #{rank}" in t and "Ground truth" in t


@pytest.mark.parametrize("budget", [0.0005, 0.01, 0.05])
@pytest.mark.parametrize("costs", [(1, 100), (1000, 1_000_000)])
def test_simulator_extreme_inputs(budget, costs, results):
    at = _goto(_app(), PAGES[3])
    at.select_slider[0].set_value(budget).run()
    at.number_input[0].set_value(costs[0]).run()
    at.number_input[1].set_value(costs[1]).run()
    assert not at.exception
    t = _text(at)
    assert "nan" not in t.lower() and "inf" not in t.lower()
    if budget == 0.01:
        rec = results["test_metrics"]["LightGBM monotone (champion)"]["recall_at_1pct"]
        assert f"{100 * rec:.1f}%" in t                     # simulator agrees with the evaluation report


def test_investigate_selection_sticks_on_repeated_changes():
    """Regression: every second selectbox change used to be silently discarded."""
    at = _goto(_app(), PAGES[2])
    for rank in (434, 2500, 7, 1200, 2):
        at.selectbox[0].set_value(rank).run()
        assert at.selectbox[0].value == rank and f"Alert #{rank}" in _text(at)


def test_selected_alert_survives_page_switches():
    at = _goto(_app(), PAGES[2])
    at.selectbox[0].set_value(1200).run()
    _goto(at, PAGES[0]); _goto(at, PAGES[2])
    assert at.selectbox[0].value == 1200
