"""AML scoring API.

Run:   uv run uvicorn api.main:app --port 8000
Docs:  http://127.0.0.1:8000/docs   (interactive Swagger UI)

Input = the transaction's model features (produced by aml.features from the
account's transaction history). Output = calibrated probability, risk band and
the top-3 plain-English reasons (SHAP).
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aml.config import PROJECT_ROOT
from aml.explain import load_model, prepare, reason_codes

app = FastAPI(title="AML Transaction Monitoring API", version="1.0.0",
              description="Scores bank transfers for money-laundering risk with a monotone LightGBM model, "
                          "calibrated probabilities and SHAP reason codes.")


@lru_cache(maxsize=1)
def serving() -> dict:
    return json.loads((PROJECT_ROOT / "models" / "serving.json").read_text())


class Transaction(BaseModel):
    features: dict[str, Union[float, int, str]] = Field(
        ..., description="Model feature name -> value (see GET /model for the list).")


class Reason(BaseModel):
    feature: str
    contribution: float
    reason: str


class Score(BaseModel):
    probability: float = Field(..., description="Calibrated P(laundering)")
    probability_uncalibrated: float
    risk_band: str = Field(..., description="High / Medium / Low")
    reasons: list[Reason]


def _band(p: float) -> str:
    b = serving()["bands"]
    return "High" if p >= b["high"] else "Medium" if p >= b["medium"] else "Low"


def _score_frame(df: pd.DataFrame) -> list[Score]:
    _, meta = load_model()
    missing = [f for f in meta["features"] if f not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {missing}")
    try:
        X = prepare(df)
        for f in meta["features"]:
            if f not in meta["categories"]:
                X[f] = pd.to_numeric(X[f], errors="raise")
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"invalid feature value: {e}")
    rc = reason_codes(X)
    c = serving()["calibrator"]
    out = []
    for e in rc:
        p_cal = float(min(np.interp(e["probability"], c["x"], c["y"]), c.get("max_probability", 1.0)))
        out.append(Score(probability=p_cal, probability_uncalibrated=e["probability"], risk_band=_band(p_cal),
                         reasons=[Reason(feature=r["feature"], contribution=r["contribution"], reason=r["reason"])
                                  for r in e["reasons"]]))
    return out


@app.get("/health")
def health():
    load_model(); serving()
    return {"status": "ok"}


@app.get("/model")
def model_info():
    _, meta = load_model()
    return {"champion": meta["champion"], "features": meta["features"], "categories": meta["categories"],
            "monotone_constraints": meta["monotone_constraints"], "risk_bands": serving()["bands"],
            "band_meaning": serving()["band_meaning"]}


@app.post("/score", response_model=Score)
def score(tx: Transaction):
    return _score_frame(pd.DataFrame([tx.features]))[0]


@app.post("/score/batch", response_model=list[Score])
def score_batch(txs: list[Transaction]):
    if not txs:
        return []
    if len(txs) > 1000:
        raise HTTPException(status_code=413, detail="max 1000 transactions per request")
    return _score_frame(pd.DataFrame([t.features for t in txs]))
