"""API tests with FastAPI's TestClient (skipped if model/serving files are absent)."""

import json

import pandas as pd
import pytest

from aml.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (PROJECT_ROOT / "models" / "serving.json").exists() or
                                not (PROJECT_ROOT / "app" / "data" / "alerts.parquet").exists(),
                                reason="serving artefacts not built")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def examples():
    meta = json.loads((PROJECT_ROOT / "models" / "model_meta.json").read_text())
    a = pd.read_parquet(PROJECT_ROOT / "app" / "data" / "alerts.parquet")
    rows = a[meta["features"]].head(5)
    return [{k: (v if isinstance(v, str) else float(v)) for k, v in r.items()} for r in rows.to_dict("records")], a.head(5)


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    m = client.get("/model").json()
    assert m["champion"] == "lgbm_monotone" and len(m["features"]) == 38


def test_score_matches_precomputed_dashboard_scores(client, examples):
    feats, alerts = examples
    for f, (_, row) in zip(feats, alerts.iterrows()):
        r = client.post("/score", json={"features": f}).json()
        assert r["probability"] == pytest.approx(row["p_cal"], rel=1e-6)
        assert r["risk_band"] == row["band"]
        assert 1 <= len(r["reasons"]) <= 3 and r["reasons"][0]["reason"] == row["reason_1"]


def test_batch_and_validation(client, examples):
    feats, _ = examples
    r = client.post("/score/batch", json=[{"features": f} for f in feats])
    assert r.status_code == 200 and len(r.json()) == 5
    bad = dict(feats[0]); bad.pop("is_ach")
    assert client.post("/score", json={"features": bad}).status_code == 422
    bad2 = dict(feats[0]); bad2["s_out_cnt_1h"] = "lots"
    assert client.post("/score", json={"features": bad2}).status_code == 422
