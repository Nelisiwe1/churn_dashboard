"""Customer Churn Dashboard — Streamlit app.

Run:  streamlit run app.py
Needs telco_churn_clean.csv (from Stage 1 of the notebook) in the same folder.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Churn Dashboard", page_icon="📉", layout="wide")

DATA = Path(__file__).parent / "telco_churn_clean.csv"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
STAYED, CHURNED, MUTED = "#2a78d6", "#eb6834", "#8a8984"
TIERS = [(0.3, "Low"), (0.5, "Medium"), (0.7, "High"), (1.01, "Very high")]


# ---------- Model: same recipe as the notebook (balanced logistic regression) ----------
@st.cache_resource
def train():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["customerID", "Churn"])
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_cols),
    ])
    model = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(X, df["Churn"])

    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    orig_of = {f: next(c for c in X.columns if f == c or f.startswith(c + "_")) for f in names}
    Z = prep.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    df["churn_prob"] = model.predict_proba(X)[:, 1]
    return df, X.columns.tolist(), model, names, orig_of, Z.mean(axis=0)


df, FEATURES, model, NAMES, ORIG_OF, Z_MEAN = train()
COEF = model.named_steps["model"].coef_[0]


def tier(p):
    return next(label for cut, label in TIERS if p < cut)


def contributions(rows: pd.DataFrame) -> pd.DataFrame:
    """Each feature's push on risk vs an average customer (log-odds), one column per original feature."""
    Z = model.named_steps["prep"].transform(rows[FEATURES])
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    raw = pd.DataFrame((Z - Z_MEAN) * COEF, columns=NAMES, index=rows.index)
    return raw.T.groupby(ORIG_OF).sum().T


def reasons(row, contrib, k=3):
    c = contrib[contrib > 0].sort_values(ascending=False).head(k)
    return "; ".join(f"{col}={row[col]}" for col in c.index)


current = df[df["Churn"] == 0].copy()
current["risk_tier"] = current["churn_prob"].map(tier)


# ---------- Header ----------
st.title("Customer Churn Dashboard")
st.caption("Which customers are most likely to leave, and why? · IBM Telco dataset · balanced logistic regression (ROC-AUC ≈ 0.84)")

tab_predict, tab_list, tab_overview = st.tabs(["🔮 Predict a customer", "📋 At-risk customers", "📊 Churn drivers"])


# ---------- Tab 1: single-customer prediction ----------
with tab_predict:
    st.subheader("Enter a customer's details")
    opts = {c: sorted(df[c].unique()) for c in FEATURES if c not in NUM_COLS}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Account**")
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        contract = st.selectbox("Contract", opts["Contract"], index=opts["Contract"].index("Month-to-month"))
        payment = st.selectbox("Payment method", opts["PaymentMethod"], index=opts["PaymentMethod"].index("Electronic check"))
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 85.0, step=5.0)
        total = st.number_input("Total charges ($)", 0.0, 9000.0, round(monthly * tenure, 2),
                                help="Defaults to tenure × monthly charge")
    with c2:
        st.markdown("**Services**")
        internet = st.selectbox("Internet service", opts["InternetService"], index=opts["InternetService"].index("Fiber optic"))
        phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
        has_net = internet != "No"
        addons = {}
        for col, label in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                           ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
                           ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]:
            addons[col] = "Yes" if st.checkbox(label, disabled=not has_net) and has_net else "No"
        multi = "Yes" if st.checkbox("Multiple lines", disabled=phone == "No") and phone == "Yes" else "No"
    with c3:
        st.markdown("**Demographics**")
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)

    customer = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
        **addons, "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }])[FEATURES]

    prob = model.predict_proba(customer)[0, 1]
    contrib = contributions(customer).iloc[0]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn risk score", f"{prob:.0%}")
    m2.metric("Risk tier", tier(prob))
    m3.metric("Ranks riskier than", f"{(current['churn_prob'] < prob).mean():.0%} of current customers")

    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8]).reset_index()
    top.columns = ["feature", "push"]
    top["label"] = [f"{f} = {customer.iloc[0][f]}" for f in top["feature"]]
    top["direction"] = np.where(top["push"] > 0, "Raises risk", "Lowers risk")
    chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("push:Q", title="Push on risk vs an average customer (log-odds)", axis=alt.Axis(tickCount=6)),
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelFontSize=12)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[CHURNED, STAYED]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip("push:Q", format="+.2f", title="Push")],
    ).properties(height=alt.Step(34), title="Why this score?")
    st.altair_chart(chart, width="stretch")
    st.caption("Scores are best read as a ranking: the model was trained with balanced class weights, so raw probabilities run high.")


# ---------- Tab 2: ranked list of current customers ----------
with tab_list:
    st.subheader("Current customers ranked by churn risk")
    f1, f2, f3 = st.columns([2, 2, 1])
    tiers = f1.multiselect("Risk tier", ["Very high", "High", "Medium", "Low"], default=["Very high", "High"])
    contracts = f2.multiselect("Contract", opts["Contract"], default=opts["Contract"])
    n_show = f3.number_input("Rows", 10, 5000, 50, step=10)

    view = current[current["risk_tier"].isin(tiers) & current["Contract"].isin(contracts)] \
        .sort_values("churn_prob", ascending=False)

    k1, k2, k3 = st.columns(3)
    k1.metric("Customers in selection", f"{len(view):,}")
    k2.metric("Monthly revenue at risk", f"${view['MonthlyCharges'].sum():,.0f}")
    k3.metric("Average risk score", f"{view['churn_prob'].mean():.0%}" if len(view) else "–")

    shown = view.head(int(n_show)).copy()
    if len(shown):
        contrib_all = contributions(shown)
        shown["top_reasons"] = [reasons(shown.loc[i], contrib_all.loc[i]) for i in shown.index]
    cols = ["customerID", "churn_prob", "risk_tier", "tenure", "Contract", "InternetService",
            "PaymentMethod", "MonthlyCharges", "top_reasons"]
    st.dataframe(
        shown.reindex(columns=cols), hide_index=True, width="stretch",
        column_config={
            "customerID": "Customer",
            "churn_prob": st.column_config.ProgressColumn("Risk score", format="percent", min_value=0, max_value=1),
            "risk_tier": "Tier", "tenure": "Tenure (m)", "InternetService": "Internet", "PaymentMethod": "Payment",
            "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
            "top_reasons": st.column_config.TextColumn("Top reasons", width="large"),
        },
    )
    st.download_button("Download this list (CSV)", shown.reindex(columns=cols).to_csv(index=False),
                       "at_risk_customers.csv", "text/csv")


# ---------- Tab 3: churn drivers ----------
with tab_overview:
    overall = df["Churn"].mean()
    o1, o2, o3 = st.columns(3)
    o1.metric("Customers", f"{len(df):,}")
    o2.metric("Churn rate", f"{overall:.1%}")
    o3.metric("Current customers at High / Very high risk", f"{current['risk_tier'].isin(['High', 'Very high']).sum():,}")

    feature = st.selectbox("Churn rate by", ["Contract", "InternetService", "PaymentMethod", "tenure group",
                                              "TechSupport", "OnlineSecurity", "SeniorCitizen", "PaperlessBilling",
                                              "Partner", "Dependents", "gender"])
    data = df.assign(**{"tenure group": pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72],
                                               labels=["0–6 m", "7–12 m", "1–2 yrs", "2–4 yrs", "4–6 yrs"]).astype(str)})
    rates = data.groupby(feature, observed=True)["Churn"].agg(rate="mean", customers="size").reset_index()
    rates[feature] = rates[feature].astype(str)
    order = rates[feature].tolist() if feature == "tenure group" else rates.sort_values("rate", ascending=False)[feature].tolist()

    base = alt.Chart(rates).encode(y=alt.Y(f"{feature}:N", sort=order, title=None,
                                          axis=alt.Axis(labelLimit=250, labelFontSize=12)))
    bars = base.mark_bar(color=CHURNED, cornerRadiusEnd=4).encode(
        x=alt.X("rate:Q", axis=alt.Axis(format="%", tickCount=5), title="Churn rate"),
        tooltip=[alt.Tooltip(f"{feature}:N"), alt.Tooltip("rate:Q", format=".1%", title="Churn rate"),
                 alt.Tooltip("customers:Q", format=",", title="Customers")])
    labels = base.mark_text(align="left", dx=4, color="#52514e").encode(x="rate:Q", text=alt.Text("rate:Q", format=".0%"))
    avg = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((bars + labels + avg).properties(height=alt.Step(40),
                    title=f"Churn rate by {feature} (dashed line = overall {overall:.1%})"), width="stretch")
