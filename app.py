"""Customer Retention Intelligence API.

Run locally:  uvicorn app:app --reload
Docs page:    http://127.0.0.1:8000/docs
"""
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------- Load models ----------
cal = joblib.load("churn_model_calibrated.joblib")  # calibrated: probabilities, revenue
best = joblib.load("churn_model.joblib")            # raw XGBoost pipeline: SHAP reasons
pre = best.named_steps["pre"]
explainer = shap.TreeExplainer(best.named_steps["clf"])
NAMES = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]

app = FastAPI(title="Customer Retention Intelligence API", version="1.0")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


COLUMNS = list(Customer.model_fields.keys())


class SimRequest(BaseModel):
    customer: Customer
    changes: Dict[str, object]


# ---------- Helpers ----------
def to_df(customers: List[Customer]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in customers])[COLUMNS]


def risk_level(p: float) -> str:
    if p <= 0.25:
        return "Low"
    if p <= 0.50:
        return "Medium"
    return "High"


def top_reasons(df: pd.DataFrame, n: int = 3) -> List[List[str]]:
    xt = pre.transform(df)
    xt = xt.toarray() if hasattr(xt, "toarray") else xt
    sv = explainer.shap_values(xt)
    return [[NAMES[j] for j in np.argsort(-row)[:n] if row[j] > 0] for row in sv]


def _m2m(d):
    return d.Contract == "Month-to-month"


# Each scenario returns a modified copy of the customers
SCENARIOS = {
    "12-month plan with 10% off": lambda d: d.assign(
        MonthlyCharges=np.where(_m2m(d), d.MonthlyCharges * 0.9, d.MonthlyCharges),
        Contract=np.where(_m2m(d), "One year", d.Contract)),
    "Cheaper bundle / loyalty credit (15% off)": lambda d: d.assign(
        MonthlyCharges=np.where(d.MonthlyCharges > 80, d.MonthlyCharges * 0.85, d.MonthlyCharges)),
    "Free tech support": lambda d: d.assign(
        TechSupport=np.where(d.InternetService == "No", d.TechSupport, "Yes")),
    "Free online security": lambda d: d.assign(
        OnlineSecurity=np.where(d.InternetService == "No", d.OnlineSecurity, "Yes")),
    "Autopay with bill credit": lambda d: d.assign(
        PaymentMethod=np.where(d.PaymentMethod == "Electronic check",
                               "Bank transfer (automatic)", d.PaymentMethod)),
}


def ranked_actions(df: pd.DataFrame, probs: np.ndarray, top: int = 2):
    gains = pd.DataFrame({k: probs - cal.predict_proba(f(df))[:, 1] for k, f in SCENARIOS.items()})
    out = []
    for i, row in gains.iterrows():
        if risk_level(probs[i]) == "Low":
            out.append([])
            continue
        picks = row.nlargest(top)
        out.append([{"action": k, "estimated_risk_reduction_pts": round(float(v) * 100, 1)}
                    for k, v in picks.items() if v > 0.01])
    return out


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = to_df([customer])
    p = float(cal.predict_proba(df)[0, 1])
    return {
        "churn_probability": round(p, 4),
        "risk_level": risk_level(p),
        "monthly_revenue_at_risk": round(p * customer.MonthlyCharges, 2),
        "top_reasons": top_reasons(df)[0],
    }


@app.post("/recommend")
def recommend(customer: Customer):
    df = to_df([customer])
    probs = cal.predict_proba(df)[:, 1]
    actions = ranked_actions(df, probs)[0]
    return {
        "churn_probability": round(float(probs[0]), 4),
        "risk_level": risk_level(float(probs[0])),
        "recommended_actions": actions or "Monitor only",
        "note": "Impacts are model estimates, not proven causal effects.",
    }


@app.post("/simulate")
def simulate(req: SimRequest):
    bad = [k for k in req.changes if k not in COLUMNS]
    if bad:
        raise HTTPException(status_code=422, detail=f"Unknown fields: {bad}")
    base = req.customer.model_dump()
    new = {**base, **req.changes}
    before = float(cal.predict_proba(pd.DataFrame([base])[COLUMNS])[0, 1])
    after = float(cal.predict_proba(pd.DataFrame([new])[COLUMNS])[0, 1])
    return {
        "before": round(before, 4),
        "after": round(after, 4),
        "change_pts": round((after - before) * 100, 1),
        "note": "Estimated impact from the model, not a guaranteed outcome.",
    }


@app.post("/revenue-at-risk")
def revenue_at_risk(customers: List[Customer]):
    df = to_df(customers)
    df["churn_prob"] = cal.predict_proba(df)[:, 1]
    df["rev_at_risk"] = df["churn_prob"] * df["MonthlyCharges"]
    top = df.sort_values("rev_at_risk", ascending=False).head(10)
    return {
        "customers": len(df),
        "monthly_revenue_at_risk": round(float(df.rev_at_risk.sum()), 2),
        "annual_revenue_at_risk": round(float(df.rev_at_risk.sum()) * 12, 2),
        "by_contract": df.groupby("Contract").rev_at_risk.sum().round(2).to_dict(),
        "top_10": top[["Contract", "MonthlyCharges", "churn_prob", "rev_at_risk"]].round(3).to_dict("records"),
    }
