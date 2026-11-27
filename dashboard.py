"""Customer Retention Intelligence dashboard. Run: streamlit run dashboard.py"""
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

if not os.path.exists("churn_model_calibrated.joblib"):
    import train  # noqa: F401  (builds the model files on first run)

from app import COLUMNS, cal, ranked_actions, risk_level, top_reasons

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

st.set_page_config(page_title="Retention Intelligence", layout="wide")


@st.cache_data
def load():
    df = pd.read_csv(URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    out = df[["customerID"] + COLUMNS].copy()
    p = cal.predict_proba(out[COLUMNS])[:, 1]
    out["churn_prob"] = p
    out["risk_level"] = [risk_level(v) for v in p]
    out["rev_at_risk"] = p * out["MonthlyCharges"]
    return out


data = load()
st.title("Customer Retention Intelligence")
tab1, tab2, tab3 = st.tabs(["Overview", "At-risk customers", "What-if simulator"])

with tab1:
    st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #162235, #1D3048);
        border: 1px solid #29415C;
        padding: 22px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricLabel"] {
        color: #A9BED3;
        font-size: 14px;
    }
    div[data-testid="stMetricValue"] {
        color: #F1F5F9;
        font-size: 27px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL CUSTOMERS", f"{len(data):,}")
    c2.metric("AVERAGE CHURN RISK", f"{data.churn_prob.mean():.1%}")
    c3.metric("MONTHLY REVENUE AT RISK", f"${data.rev_at_risk.sum():,.0f}")
    c4.metric("ANNUAL REVENUE AT RISK", f"${data.rev_at_risk.sum() * 12:,.0f}")

    left, right = st.columns(2)

with left:
    st.subheader("Revenue at Risk by Contract")
    contract_data = (
        data.groupby("Contract", as_index=False)["rev_at_risk"]
        .sum()
        .sort_values("rev_at_risk", ascending=False)
    )

    fig1 = px.bar(
        contract_data,
        x="Contract",
        y="rev_at_risk",
        text_auto=",.0f",
        title=None,
        labels={
            "Contract": "Contract Type",
            "rev_at_risk": "Monthly Revenue at Risk ($)"
        },
        template="plotly_dark"
    )
    fig1.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Customers by Risk Level")
    risk_data = (
        data["risk_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .rename_axis("Risk Level")
        .reset_index(name="Customers")
    )

    fig2 = px.bar(
        risk_data,
        x="Risk Level",
        y="Customers",
        text="Customers",
        category_orders={"Risk Level": ["Low", "Medium", "High"]},
        template="plotly_dark"
    )
    fig2.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)
with tab2:
    f1, f2 = st.columns(2)
    levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    all_contracts = sorted(data.Contract.unique())
    contracts = f2.multiselect("Contract", all_contracts, default=all_contracts)
    view = data[data.risk_level.isin(levels) & data.Contract.isin(contracts)]
    view = view.sort_values("rev_at_risk", ascending=False)
    st.caption(f"{len(view):,} customers | ${view.rev_at_risk.sum():,.0f} per month at risk")
    cols = ["customerID", "Contract", "tenure", "MonthlyCharges", "churn_prob", "risk_level", "rev_at_risk"]
    st.dataframe(view[cols].head(200), use_container_width=True, hide_index=True)

    if len(view):
        cid = st.selectbox("Inspect a customer", view.customerID.head(200))
        row = data[data.customerID == cid]
        x1 = row[COLUMNS].reset_index(drop=True)
        p = float(row.churn_prob.iloc[0])
        st.markdown(f"**Churn risk:** {p:.1%} ({risk_level(p)})")
        reasons = top_reasons(x1)[0]
        st.markdown("**Main drivers:** " + (", ".join(reasons) if reasons else "none flagged"))
        acts = ranked_actions(x1, np.array([p]))[0]
        if acts:
            st.markdown("**Recommended actions (model-estimated impact):**")
            for a in acts:
                st.write(f"- {a['action']}: about {a['estimated_risk_reduction_pts']} points lower risk")
        else:
            st.markdown("**Recommended actions:** Monitor only")

with tab3:
    ranked = data.sort_values("rev_at_risk", ascending=False).customerID.head(200)
    cid = st.selectbox("Customer", ranked, key="sim")
    base = data[data.customerID == cid][COLUMNS].iloc[0].to_dict()

    def opts(col):
        o = sorted(data[col].unique())
        return o, o.index(base[col])

    a, b = st.columns(2)
    charges = a.slider("Monthly charges", 15.0, 125.0, float(base["MonthlyCharges"]))
    o, i = opts("Contract")
    contract = a.selectbox("Contract", o, index=i)
    o, i = opts("PaymentMethod")
    pay = a.selectbox("Payment method", o, index=i)
    o, i = opts("TechSupport")
    tech = b.selectbox("Tech support", o, index=i)
    o, i = opts("OnlineSecurity")
    sec = b.selectbox("Online security", o, index=i)

    new = {**base, "MonthlyCharges": charges, "Contract": contract,
           "PaymentMethod": pay, "TechSupport": tech, "OnlineSecurity": sec}
    p0 = float(cal.predict_proba(pd.DataFrame([base])[COLUMNS])[0, 1])
    p1 = float(cal.predict_proba(pd.DataFrame([new])[COLUMNS])[0, 1])
    m1, m2 = st.columns(2)
    m1.metric("Current risk", f"{p0:.1%}")
    m2.metric("Simulated risk", f"{p1:.1%}", f"{(p1 - p0) * 100:+.1f} pts", delta_color="inverse")
    st.caption("Estimated impact from the model, not a guaranteed outcome.")
