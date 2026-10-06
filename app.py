import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ChurnPredict • Customer Retention AI",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM UI
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,.10), transparent 25%),
            radial-gradient(circle at 90% 20%, rgba(236,72,153,.08), transparent 25%),
            #080b14;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1220 0%, #080b14 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    .hero {
        padding: 28px 30px;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(99,102,241,.20), rgba(168,85,247,.10));
        border: 1px solid rgba(255,255,255,.10);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 42px;
        margin: 0;
        font-weight: 800;
        letter-spacing: -1.5px;
    }

    .hero p {
        color: #aeb8cc;
        font-size: 16px;
        margin-top: 8px;
    }

    .card {
        padding: 20px;
        border-radius: 18px;
        background: rgba(17,24,39,.72);
        border: 1px solid rgba(255,255,255,.08);
        min-height: 120px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: .7px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 5px;
    }

    .risk-high {
        padding: 25px;
        border-radius: 20px;
        background: rgba(239,68,68,.12);
        border: 1px solid rgba(239,68,68,.35);
        text-align: center;
    }

    .risk-low {
        padding: 25px;
        border-radius: 20px;
        background: rgba(34,197,94,.10);
        border: 1px solid rgba(34,197,94,.30);
        text-align: center;
    }

    .risk-medium {
        padding: 25px;
        border-radius: 20px;
        background: rgba(245,158,11,.10);
        border: 1px solid rgba(245,158,11,.30);
        text-align: center;
    }

    .risk-number {
        font-size: 52px;
        font-weight: 900;
        line-height: 1;
    }

    .recommendation {
        padding: 14px 16px;
        border-radius: 13px;
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.07);
        margin: 8px 0;
    }

    .small-note {
        color: #8f9bb1;
        font-size: 12px;
    }

    div[data-testid="stMetric"] {
        background: rgba(17,24,39,.72);
        border: 1px solid rgba(255,255,255,.08);
        padding: 14px;
        border-radius: 16px;
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"
DATA_PATH = BASE_DIR / "customer_churn_dataset-testing-master.csv"

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)

def prepare_customer(data):
    """Match the preprocessing used in the Colab notebook."""
    df = pd.DataFrame([data])

    categorical_columns = [
        "Gender",
        "Subscription Type",
        "Contract Length"
    ]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    # Reproduce the notebook's feature order.
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    return df

def get_risk(probability):
    if probability >= 0.70:
        return "HIGH", "risk-high", "🔴"
    elif probability >= 0.40:
        return "MEDIUM", "risk-medium", "🟠"
    return "LOW", "risk-low", "🟢"

def recommendations(customer, probability):
    tips = []

    if probability >= 0.70:
        tips.append("🚨 <b>Immediate retention call:</b> Contact the customer with a personalized retention offer.")
    elif probability >= 0.40:
        tips.append("⚠️ <b>Monitor closely:</b> Add the customer to a proactive engagement campaign.")
    else:
        tips.append("💚 <b>Maintain relationship:</b> Continue normal engagement and loyalty activities.")

    if customer["Support Calls"] >= 5:
        tips.append("🎧 <b>Support attention:</b> High support-call activity may indicate unresolved issues.")

    if customer["Payment Delay"] >= 15:
        tips.append("💳 <b>Payment follow-up:</b> Consider a payment reminder or flexible payment option.")

    if customer["Last Interaction"] >= 20:
        tips.append("📩 <b>Re-engage:</b> The customer has not interacted recently. Send a targeted campaign.")

    if customer["Tenure"] <= 6:
        tips.append("🌱 <b>Onboarding:</b> Newer customers may benefit from onboarding and product education.")

    if customer["Usage Frequency"] <= 8:
        tips.append("📈 <b>Increase usage:</b> Recommend useful features or a personalized product walkthrough.")

    if not tips:
        tips.append("✨ <b>Healthy profile:</b> No major warning indicators detected.")

    return tips

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = load_model()
data = load_data()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📉 ChurnPredict")
    st.caption("Customer Retention Intelligence")

    st.divider()

    page = st.radio(
        "Navigate",
        ["🔮 Predict Churn", "📊 Dataset Explorer", "🧠 About the Model"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### Model status")
    if model is not None:
        st.success("Random Forest loaded")
    else:
        st.error("Model file missing")

   

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div style="font-size:14px;color:#a5b4fc;font-weight:700;">AI-POWERED RETENTION PLATFORM</div>
    <h1>ChurnPredict</h1>
    <p>Predict which customers are at risk of leaving — and turn predictions into retention actions.</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error(
        "⚠️ `churn_model.pkl` was not found. "
        "Export your trained Random Forest model from Colab and place it in the same folder as `app.py`."
    )
    st.code("""
import joblib

joblib.dump(model, "churn_model.pkl")
print("churn_model.pkl created successfully!")
""", language="python")
    st.stop()

# ---------------------------------------------------------
# PREDICT PAGE
# ---------------------------------------------------------
if page == "🔮 Predict Churn":

    st.markdown("### 🎯 Customer Risk Scanner")
    st.caption("Enter customer information and let the trained Random Forest model estimate churn probability.")

    # Quick demo profiles
    st.markdown("#### ⚡ Try a profile")

    demo1, demo2, demo3 = st.columns(3)

    if "customer" not in st.session_state:
        st.session_state.customer = {
            "Age": 25,
            "Gender": "Female",
            "Tenure": 5,
            "Usage Frequency": 20,
            "Support Calls": 3,
            "Payment Delay": 2,
            "Subscription Type": "Premium",
            "Contract Length": "Annual",
            "Total Spend": 5000.0,
            "Last Interaction": 10
        }

    with demo1:
        if st.button("🟢 Loyal Customer", use_container_width=True):
            st.session_state.customer = {
                "Age": 34, "Gender": "Female", "Tenure": 36,
                "Usage Frequency": 25, "Support Calls": 1,
                "Payment Delay": 0, "Subscription Type": "Premium",
                "Contract Length": "Annual", "Total Spend": 12000.0,
                "Last Interaction": 3
            }
            st.rerun()

    with demo2:
        if st.button("🟠 At-Risk Customer", use_container_width=True):
            st.session_state.customer = {
                "Age": 29, "Gender": "Male", "Tenure": 9,
                "Usage Frequency": 10, "Support Calls": 5,
                "Payment Delay": 14, "Subscription Type": "Standard",
                "Contract Length": "Monthly", "Total Spend": 3200.0,
                "Last Interaction": 18
            }
            st.rerun()

    with demo3:
        if st.button("🔴 High-Risk Customer", use_container_width=True):
            st.session_state.customer = {
                "Age": 22, "Gender": "Female", "Tenure": 3,
                "Usage Frequency": 4, "Support Calls": 8,
                "Payment Delay": 28, "Subscription Type": "Basic",
                "Contract Length": "Monthly", "Total Spend": 650.0,
                "Last Interaction": 30
            }
            st.rerun()

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 👤 Customer Profile")

        age = st.number_input("Age", 18, 100, int(st.session_state.customer["Age"]))
        gender = st.selectbox(
            "Gender", ["Female", "Male"],
            index=["Female", "Male"].index(st.session_state.customer["Gender"])
        )
        tenure = st.number_input("Tenure (months)", 0, 120, int(st.session_state.customer["Tenure"]))
        usage = st.number_input("Usage Frequency", 0, 100, int(st.session_state.customer["Usage Frequency"]))
        support = st.number_input("Support Calls", 0, 50, int(st.session_state.customer["Support Calls"]))

    with c2:
        st.markdown("#### 💳 Subscription & Activity")

        payment = st.number_input("Payment Delay (days)", 0, 100, int(st.session_state.customer["Payment Delay"]))
        subscription = st.selectbox(
            "Subscription Type",
            ["Basic", "Standard", "Premium"],
            index=["Basic", "Standard", "Premium"].index(st.session_state.customer["Subscription Type"])
        )
        contract = st.selectbox(
            "Contract Length",
            ["Monthly", "Quarterly", "Annual"],
            index=["Monthly", "Quarterly", "Annual"].index(st.session_state.customer["Contract Length"])
        )
        spend = st.number_input(
            "Total Spend",
            min_value=0.0,
            max_value=1000000.0,
            value=float(st.session_state.customer["Total Spend"]),
            step=100.0
        )
        interaction = st.number_input(
            "Last Interaction (days)",
            0, 365,
            int(st.session_state.customer["Last Interaction"])
        )

    customer = {
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support,
        "Payment Delay": payment,
        "Subscription Type": subscription,
        "Contract Length": contract,
        "Total Spend": spend,
        "Last Interaction": interaction
    }

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 ANALYZE CUSTOMER RISK", type="primary", use_container_width=True):
        try:
            customer_df = prepare_customer(customer)

            prediction = int(model.predict(customer_df)[0])
            probability = float(model.predict_proba(customer_df)[0][1])

            st.session_state.result = {
                "prediction": prediction,
                "probability": probability,
                "customer": customer
            }

        except Exception as e:
            st.error("Prediction could not be completed.")
            st.exception(e)

    if "result" in st.session_state:
        result = st.session_state.result
        probability = result["probability"]
        prediction = result["prediction"]
        customer = result["customer"]

        st.divider()
        st.markdown("## 📡 AI Risk Report")

        risk, risk_class, icon = get_risk(probability)

        left, right = st.columns([1, 1.5])

        with left:
            st.markdown(
                f"""
                <div class="{risk_class}">
                    <div style="font-size:18px;font-weight:700;">{icon} {risk} RISK</div>
                    <div class="risk-number">{probability:.0%}</div>
                    <div style="color:#aeb8cc;margin-top:8px;">Predicted churn probability</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            if prediction == 1:
                st.error("🔴 **Prediction: Customer is likely to CHURN**")
            else:
                st.success("🟢 **Prediction: Customer is likely to STAY**")

            st.progress(probability)
            st.caption(f"Model confidence indicator: {probability:.1%}")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Tenure", f"{customer['Tenure']} mo")
        with m2:
            st.metric("Support Calls", customer["Support Calls"])
        with m3:
            st.metric("Payment Delay", f"{customer['Payment Delay']} days")
        with m4:
            st.metric("Usage", customer["Usage Frequency"])

        st.markdown("### 💡 Recommended Actions")

        for tip in recommendations(customer, probability):
            st.markdown(
                f'<div class="recommendation">{tip}</div>',
                unsafe_allow_html=True
            )

        with st.expander("🔎 View model input"):
            st.dataframe(pd.DataFrame([customer]), use_container_width=True)

# ---------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------
elif page == "📊 Dataset Explorer":

    st.markdown("### 📊 Customer Dataset Explorer")

    if data is None:
        st.warning("Dataset CSV not found in the project folder.")
    else:
        total = len(data)
        churned = int(data["Churn"].sum())
        churn_rate = churned / total if total else 0

        a, b, c, d = st.columns(4)

        with a:
            st.metric("Customers", f"{total:,}")
        with b:
            st.metric("Churned", f"{churned:,}")
        with c:
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        with d:
            st.metric("Avg Spend", f"{data['Total Spend'].mean():,.0f}")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Churn distribution")
            churn_counts = data["Churn"].value_counts().rename(index={0: "Stayed", 1: "Churned"})
            st.bar_chart(churn_counts)

        with col2:
            st.markdown("#### Average customer behavior")
            avg = data.groupby("Churn")[[
                "Tenure", "Usage Frequency", "Support Calls",
                "Payment Delay", "Total Spend"
            ]].mean()
            avg.index = ["Stayed", "Churned"]
            st.dataframe(avg.round(2), use_container_width=True)

        st.divider()

        st.markdown("#### 🔍 Search customers")
        search = st.text_input("Search by Customer ID", placeholder="Example: 1024")

        display = data.copy()

        if search:
            try:
                cid = int(search)
                display = display[display["CustomerID"] == cid]
            except ValueError:
                st.warning("Enter a numeric Customer ID.")

        st.dataframe(display, use_container_width=True, height=430)

# ---------------------------------------------------------
# MODEL PAGE
# ---------------------------------------------------------
else:

    st.markdown("### 🧠 About ChurnIQ")

    st.markdown("""
    **ChurnIQ** uses the Random Forest model you trained in your Colab notebook.

    The Streamlit app does **not retrain the model**. It loads your exported
    `churn_model.pkl` and applies the same categorical preprocessing used during training.
    """)

    st.markdown("#### 🌲 Model pipeline")

    steps = [
        ("01", "Customer Data", "Age, tenure, usage, support, payment and subscription information"),
        ("02", "Preprocessing", "Categorical fields are converted using pandas get_dummies()"),
        ("03", "Random Forest", "100 decision trees evaluate customer behavior"),
        ("04", "Probability", "The model estimates the probability that the customer will churn"),
        ("05", "Action", "The dashboard converts the prediction into retention recommendations")
    ]

    for num, title, desc in steps:
        st.markdown(
            f"""
            <div class="recommendation">
                <b>{num} · {title}</b><br>
                <span style="color:#94a3b8;">{desc}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("#### 📌 Input features")

    features = [
        "Age", "Gender", "Tenure", "Usage Frequency",
        "Support Calls", "Payment Delay", "Subscription Type",
        "Contract Length", "Total Spend", "Last Interaction"
    ]

    cols = st.columns(2)
    for i, feature in enumerate(features):
        cols[i % 2].markdown(f"• **{feature}**")

    st.info(
        "This dashboard is designed as a decision-support tool. "
        "A high churn probability indicates higher predicted risk; it does not guarantee that a customer will leave."
    )

st.markdown("""
<div style="text-align:center;color:#64748b;font-size:12px;padding:35px 0 10px;">
    ChurnIQ • Customer Retention Intelligence • Powered by your trained Random Forest model
</div>
""", unsafe_allow_html=True)
