"""
Streamlit app — Uplift-Based Campaign Targeting Simulator
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.uplift_metrics import expected_incremental_profit, qini_curve, qini_coefficient

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

st.set_page_config(page_title="Uplift Campaign Simulator", page_icon="🎯", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "uplift_model.joblib")
    metadata = json.load(open(MODELS_DIR / "metadata.json"))
    X_test = pd.read_csv(REPORTS_DIR / "X_test.csv")
    outcomes = pd.read_csv(REPORTS_DIR / "test_outcomes.csv")
    return model, metadata, X_test, outcomes


st.title("🎯 Uplift-Based Campaign Targeting Simulator")
st.caption(
    "Instead of predicting who will convert, this predicts who will convert "
    "**because** they were targeted — and who should be left alone."
)

try:
    model, metadata, X_test, outcomes = load_artifacts()
except FileNotFoundError:
    st.error("No trained model found. Run `python run_pipeline.py` first.")
    st.stop()

t_test = outcomes["treatment"].values
y_test = outcomes["conversion"].values
uplift_scores = model.predict_uplift(X_test)

with st.sidebar:
    st.header("Model info")
    st.write(f"**Best model:** {metadata['best_model']}")
    st.metric("Qini coefficient", f"{metadata['qini_coefficient']:.4f}")
    st.metric("Uplift @ top 30%", f"{metadata['uplift_at_30pct']:.4f}")

tab1, tab2 = st.tabs(["📊 Campaign Simulator", "🔍 Score a Single Customer"])

with tab1:
    st.subheader("How many customers should you target?")
    st.write(
        "Blasting your entire list wastes budget on **Sure Things** (would "
        "convert anyway) and **Lost Causes** (never will), and can even "
        "backfire on **Sleeping Dogs** (targeting reduces their conversion). "
        "This simulates expected profit at different targeting thresholds."
    )

    col1, col2 = st.columns(2)
    with col1:
        cost = st.number_input("Cost per customer treated ($)", value=2.50, step=0.50)
    with col2:
        value = st.number_input("Value per conversion ($)", value=45.0, step=5.0)

    target_pct = st.slider("% of customers to target (ranked by predicted uplift)", 1, 100, 30)

    result = expected_incremental_profit(
        uplift_scores, len(X_test), cost, value, target_pct / 100
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers targeted", f"{result['n_targeted']:,}")
    c2.metric("Expected incremental conversions", f"{result['expected_incremental_conversions']:.1f}")
    c3.metric("Expected net profit", f"${result['expected_net_profit']:,.0f}")

    # Sweep to show the optimal point
    fractions = np.linspace(0.05, 1.0, 20)
    profits = [
        expected_incremental_profit(uplift_scores, len(X_test), cost, value, f)["expected_net_profit"]
        for f in fractions
    ]
    best_frac = fractions[int(np.argmax(profits))]
    st.line_chart(pd.DataFrame({"% targeted": fractions * 100, "Expected profit ($)": profits}).set_index("% targeted"))
    st.info(f"💡 Estimated optimal targeting threshold: **top {best_frac*100:.0f}%** of customers.")

with tab2:
    st.subheader("Score an individual customer")
    st.write("Enter feature values (anonymized `f0`-`f11`, same schema as the source dataset).")

    cols = st.columns(4)
    feature_values = {}
    for i in range(12):
        with cols[i % 4]:
            feature_values[f"f{i}"] = st.slider(f"f{i}", -3.0, 3.0, 0.0, 0.1, key=f"f{i}")

    if st.button("Predict Uplift", type="primary"):
        X_single = pd.DataFrame([feature_values])
        score = model.predict_uplift(X_single)[0]

        st.divider()
        st.metric("Predicted individual treatment effect", f"{score:+.3f}")

        if score > 0.05:
            st.success("🎯 **Persuadable** — targeting this customer is likely to work. Include in campaign.")
        elif score < -0.02:
            st.error("🐕 **Possible Sleeping Dog** — targeting may REDUCE conversion. Exclude from campaign.")
        else:
            st.warning("😐 **Sure Thing / Lost Cause** — near-zero incremental effect. Low priority; save the budget.")
