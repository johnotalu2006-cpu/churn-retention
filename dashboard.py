
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

if not os.path.exists("churn_model_calibrated.joblib"):
    import train  # noqa: F401

from app import COLUMNS, cal, ranked_actions, risk_level, top_reasons


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# DESIGN SYSTEM
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1220;
}

[data-testid="stSidebar"] {
    background: #101B2D;
    border-right: 1px solid #24354D;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #A9BED3;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.main-title {
    font-size: 30px;
    font-weight: 800;
    color: #F1F5F9;
    margin-bottom: 4px;
}

.subtitle {
    color: #94A3B8;
    font-size: 14px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 19px;
    font-weight: 700;
    color: #F1F5F9;
    margin-top: 20px;
    margin-bottom: 12px;
}

.kpi-card {
    background: linear-gradient(145deg, #16253A, #1B304A);
    border: 1px solid #2B405B;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.kpi-label {
    color: #A9BED3;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}

.kpi-value {
    color: #F8FAFC;
    font-size: 25px;
    font-weight: 800;
}

.kpi-icon {
    font-size: 17px;
    margin-bottom: 8px;
}

.info-panel {
    background: #16253A;
    border: 1px solid #2B405B;
    border-radius: 13px;
    padding: 18px;
}

.small-muted {
    color: #94A3B8;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #16253A;
    border: 1px solid #2B405B;
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #A9BED3;
}

div[data-testid="stMetricValue"] {
    color: #F1F5F9;
    font-weight: 700;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #2B405B;
    border-radius: 10px;
}

.stButton button {
    border-radius: 9px;
}

hr {
    border-color: #26374D;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA
# --------------------------------------------------

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


@st.cache_data
def load():
    df = pd.read_csv(URL)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].str.strip(),
        errors="coerce"
    )

    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    out = df[["customerID"] + COLUMNS].copy()

    probabilities = cal.predict_proba(out[COLUMNS])[:, 1]

    out["churn_prob"] = probabilities
    out["risk_level"] = [
        risk_level(value) for value in probabilities
    ]

    out["rev_at_risk"] = (
        probabilities * out["MonthlyCharges"]
    )

    return out


data = load()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📊 RETENTION AI")
    st.caption("CUSTOMER INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Customer Explorer",
            "Retention Simulator"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**MODEL STATUS**")
    st.success("Prediction model loaded")

    st.caption("Customer Retention Intelligence")
    st.caption("Machine learning decision support")


# --------------------------------------------------
# SHARED HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Customer Retention Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Understand churn risk. Identify revenue exposure. '
    'Make smarter retention decisions.</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.markdown("### Business Overview")

    total_customers = len(data)
    average_risk = data.churn_prob.mean()
    monthly_risk = data.rev_at_risk.sum()
    annual_risk = monthly_risk * 12

    k1, k2, k3, k4 = st.columns(4)

    cards = [
        (k1, "TOTAL CUSTOMERS", f"{total_customers:,}", "👥"),
        (k2, "AVERAGE CHURN RISK", f"{average_risk:.1%}", "📉"),
        (k3, "MONTHLY REVENUE AT RISK", f"${monthly_risk:,.0f}", "💰"),
        (k4, "ANNUAL REVENUE AT RISK", f"${annual_risk:,.0f}", "📊")
    ]

    for col, label, value, icon in cards:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("")
    st.markdown("### Churn Analytics")

    left, right = st.columns(2)

    with left:
        st.markdown("**Revenue at Risk by Contract**")

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
            labels={
                "Contract": "Contract Type",
                "rev_at_risk": "Revenue at Risk ($)"
            },
            color="Contract",
            template="plotly_dark"
        )

        fig1.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with right:
        st.markdown("**Customer Risk Distribution**")

        risk_data = (
            data["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
            .rename_axis("Risk Level")
            .reset_index(name="Customers")
        )

        fig2 = px.pie(
            risk_data,
            names="Risk Level",
            values="Customers",
            hole=0.65,
            color="Risk Level",
            color_discrete_map={
                "Low": "#22C55E",
                "Medium": "#F59E0B",
                "High": "#EF4444"
            },
            template="plotly_dark"
        )

        fig2.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("### Priority Customers")

    priority = data.sort_values(
        "rev_at_risk",
        ascending=False
    ).head(10)

    st.dataframe(
        priority[
            [
                "customerID",
                "Contract",
                "tenure",
                "churn_prob",
                "risk_level",
                "rev_at_risk"
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "customerID": "Customer ID",
            "Contract": "Contract",
            "tenure": st.column_config.NumberColumn(
                "Tenure (months)"
            ),
            "churn_prob": st.column_config.ProgressColumn(
                "Churn Probability",
                format="%.1f%%",
                min_value=0,
                max_value=1
            ),
            "risk_level": "Risk Level",
            "rev_at_risk": st.column_config.NumberColumn(
                "Monthly Revenue at Risk",
                format="$%.2f"
            )
        }
    )


# --------------------------------------------------
# CUSTOMER EXPLORER
# --------------------------------------------------

elif page == "Customer Explorer":

    st.markdown("### Customer Risk Explorer")

    st.caption(
        "Identify high-risk customers and inspect their individual "
        "churn drivers and recommended retention actions."
    )

    f1, f2 = st.columns(2)

    levels = f1.multiselect(
        "Risk level",
        ["High", "Medium", "Low"],
        default=["High"]
    )

    all_contracts = sorted(data.Contract.unique())

    contracts = f2.multiselect(
        "Contract type",
        all_contracts,
        default=all_contracts
    )

    search_id = st.text_input(
        "Search customer ID",
        placeholder="Enter customer ID"
    )

    view = data[
        data.risk_level.isin(levels)
        & data.Contract.isin(contracts)
    ].copy()

    if search_id.strip():
        view = view[
            view.customerID.str.contains(
                search_id.strip(),
                case=False,
                na=False
            )
        ]

    view = view.sort_values(
        "rev_at_risk",
        ascending=False
    )

    m1, m2 = st.columns(2)

    m1.metric(
        "Matching Customers",
        f"{len(view):,}"
    )

    m2.metric(
        "Monthly Revenue at Risk",
        f"${view.rev_at_risk.sum():,.0f}"
    )

    st.markdown("### Customer List")

    st.dataframe(
        view[
            [
                "customerID",
                "Contract",
                "tenure",
                "MonthlyCharges",
                "churn_prob",
                "risk_level",
                "rev_at_risk"
            ]
        ].head(200),
        use_container_width=True,
        hide_index=True,
        column_config={
            "customerID": "Customer ID",
            "Contract": "Contract",
            "tenure": "Tenure (months)",
            "MonthlyCharges": st.column_config.NumberColumn(
                "Monthly Charges",
                format="$%.2f"
            ),
            "churn_prob": st.column_config.ProgressColumn(
                "Churn Probability",
                format="%.1f%%",
                min_value=0,
                max_value=1
            ),
            "risk_level": "Risk Level",
            "rev_at_risk": st.column_config.NumberColumn(
                "Monthly Revenue at Risk",
                format="$%.2f"
            )
        }
    )

    st.divider()

    st.markdown("### Customer Inspection")

    if len(view):

        cid = st.selectbox(
            "Select a customer",
            view.customerID.head(200)
        )

        row = data[data.customerID == cid]

        x1 = row[COLUMNS].reset_index(drop=True)

        probability = float(row.churn_prob.iloc[0])

        a, b, c = st.columns(3)

        a.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

        b.metric(
            "Risk Category",
            risk_level(probability)
        )

        c.metric(
            "Monthly Charges",
            f"${float(row.MonthlyCharges.iloc[0]):,.2f}"
        )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Main Risk Drivers")

            reasons = top_reasons(x1)[0]

            if reasons:
                for reason in reasons:
                    st.write(f"• {reason}")
            else:
                st.info("No major risk drivers flagged.")

        with right:
            st.markdown("#### Recommended Retention Actions")

            actions = ranked_actions(
                x1,
                np.array([probability])
            )[0]

            if actions:
                for action in actions:
                    st.write(
                        f"• **{action['action']}** — "
                        f"estimated risk reduction: "
                        f"{action['estimated_risk_reduction_pts']} points"
                    )
            else:
                st.info("Monitor this customer.")

    else:
        st.info("No customers match the selected filters.")


# --------------------------------------------------
# RETENTION SIMULATOR
# --------------------------------------------------

elif page == "Retention Simulator":

    st.markdown("### Retention What-if Simulator")

    st.caption(
        "Explore how changes to a customer's subscription and "
        "services may influence their predicted churn risk."
    )

    ranked = data.sort_values(
        "rev_at_risk",
        ascending=False
    ).customerID.head(200)

    cid = st.selectbox(
        "Select customer",
        ranked,
        key="sim"
    )

    base = data[
        data.customerID == cid
    ][COLUMNS].iloc[0].to_dict()

    def opts(col):
        options = sorted(data[col].unique())
        return options, options.index(base[col])

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("#### Billing and Contract")

        minimum = float(data["MonthlyCharges"].min())
        maximum = float(data["MonthlyCharges"].max())

        charges = st.slider(
            "Monthly Charges ($)",
            min_value=minimum,
            max_value=maximum,
            value=float(base["MonthlyCharges"])
        )

        options, index = opts("Contract")

        contract = st.selectbox(
            "Contract Type",
            options,
            index=index
        )

        options, index = opts("PaymentMethod")

        payment = st.selectbox(
            "Payment Method",
            options,
            index=index
        )

    with right:

        st.markdown("#### Additional Services")

        options, index = opts("TechSupport")

        tech = st.selectbox(
            "Tech Support",
            options,
            index=index
        )

        options, index = opts("OnlineSecurity")

        security = st.selectbox(
            "Online Security",
            options,
            index=index
        )

    new = {
        **base,
        "MonthlyCharges": charges,
        "Contract": contract,
        "PaymentMethod": payment,
        "TechSupport": tech,
        "OnlineSecurity": security
    }

    current_probability = float(
        cal.predict_proba(
            pd.DataFrame([base])[COLUMNS]
        )[0, 1]
    )

    simulated_probability = float(
        cal.predict_proba(
            pd.DataFrame([new])[COLUMNS]
        )[0, 1]
    )

    change = (
        simulated_probability - current_probability
    ) * 100

    st.divider()

    st.markdown("### Simulation Results")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Current Churn Risk",
        f"{current_probability:.1%}"
    )

    m2.metric(
        "Simulated Churn Risk",
        f"{simulated_probability:.1%}",
        f"{change:+.1f} percentage points",
        delta_color="inverse"
    )

    m3.metric(
        "Simulated Risk Category",
        risk_level(simulated_probability)
    ),

    if change < 0:

        st.success(
            f"Predicted churn risk decreases by "
            f"{abs(change):.1f} percentage points."
        )

    elif change > 0:

        st.warning(
            f"Predicted churn risk increases by "
            f"{change:.1f} percentage points."
        )

    else:

        st.info(
            "The simulated changes do not alter predicted churn risk."
        )

    st.caption(
        "These are model-based estimates, not guaranteed outcomes. "
        "The simulation does not establish causal effects."
    )

    st.divider()

    st.markdown("### Model Interpretation")

    st.write(
        "The simulator compares the model's prediction for the "
        "customer's original details with its prediction after "
        "the selected changes."
    )
    
