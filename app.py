"""Streamlit dashboard for manufacturing customer churn and retention analytics."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "manufacturing_customers.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Manufacturing Churn Analytics", page_icon="🏭", layout="wide")
st.title("🏭 Customer Churn Prediction & Retention Analytics")
st.caption("Portfolio prototype for a manufacturing business • Uses synthetic demonstration data")

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.warning("Project files are not ready yet. Run these commands in the VS Code terminal:")
    st.code("python generate_data.py\npython train_model.py\npython -m streamlit run app.py", language="powershell")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
else:
    metrics = {}

with st.sidebar:
    st.header("Filters")
    regions = st.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    categories = st.multiselect("Product category", sorted(df["product_category"].unique()),
                                default=sorted(df["product_category"].unique()))
    risk_threshold = st.slider("High-risk threshold (%)", 20, 90, 60, 5) / 100

filtered = df[df["region"].isin(regions) & df["product_category"].isin(categories)].copy()
FEATURES = [
    "tenure_months", "orders_last_90_days", "avg_order_value",
    "days_since_last_order", "late_delivery_rate", "complaint_count",
    "payment_delay_days", "discount_dependency", "order_frequency_change",
    "region", "product_category", "customer_type"
]
filtered["churn_probability"] = model.predict_proba(filtered[FEATURES])[:, 1]
filtered["risk_level"] = np.where(filtered["churn_probability"] >= risk_threshold, "High risk",
                          np.where(filtered["churn_probability"] >= 0.35, "Medium risk", "Low risk"))
filtered["estimated_annual_revenue"] = (
    filtered["avg_order_value"] * filtered["orders_last_90_days"] * 4
)

churn_rate = filtered["churn"].mean() if len(filtered) else 0
high_risk = filtered[filtered["churn_probability"] >= risk_threshold]
revenue_risk = high_risk["estimated_annual_revenue"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers in view", f"{len(filtered):,}")
c2.metric("Historical churn rate", f"{churn_rate:.1%}")
c3.metric("Predicted high-risk customers", f"{len(high_risk):,}")
c4.metric("Estimated annual revenue at risk", f"₹{revenue_risk:,.0f}")

st.info("Revenue at risk is a rough prioritization estimate, not a financial forecast. It annualizes recent order value and assumes purchasing patterns continue.")

left, right = st.columns(2)
with left:
    st.subheader("Historical churn by product category")
    chart_data = filtered.groupby("product_category", as_index=False)["churn"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=chart_data, x="product_category", y="churn", ax=ax)
    ax.set_ylabel("Historical churn rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Predicted risk distribution")
    risk_counts = filtered["risk_level"].value_counts().reindex(
        ["High risk", "Medium risk", "Low risk"], fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=risk_counts.index, y=risk_counts.values, ax=ax)
    ax.set_ylabel("Number of customers")
    ax.set_xlabel("")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Customers to prioritize")
st.write("Review high-risk customers first. Prediction is a prioritization aid, not proof that a customer will leave.")
columns_to_show = [
    "customer_id", "customer_type", "region", "product_category",
    "days_since_last_order", "orders_last_90_days", "late_delivery_rate",
    "complaint_count", "avg_order_value", "churn_probability", "risk_level"
]
priority = filtered[filtered["churn_probability"] >= risk_threshold].sort_values(
    "churn_probability", ascending=False
)
st.dataframe(priority[columns_to_show].style.format({
    "late_delivery_rate": "{:.0%}",
    "avg_order_value": "₹{:,.0f}",
    "churn_probability": "{:.0%}"
}), use_container_width=True, hide_index=True)

st.download_button(
    "Download prioritized customer list",
    data=priority[columns_to_show].to_csv(index=False).encode("utf-8"),
    file_name="high_risk_customers.csv",
    mime="text/csv"
)

st.subheader("Feature Importance Analysis")

# Get the trained Random Forest model
rf_model = model.named_steps["model"]

# Get the preprocessing step
preprocessor = model.named_steps["preprocess"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get importance scores
importances = rf_model.feature_importances_

# Create a DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort and select the top 10 features
importance_df = importance_df.sort_values(
    by="Importance", ascending=False
).head(10)

# Display chart
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    ax=ax
)

ax.set_title("Top 10 Features Influencing Churn Predictions")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Retention playbook")
st.markdown("""
- **Long gap since last order:** account manager checks in and asks about upcoming demand.
- **High late-delivery rate:** review delivery performance and share a corrective action plan.
- **Repeated complaints:** assign an owner and close open service issues.
- **Falling order frequency:** discuss forecast, stock needs, and replenishment schedule.
- **Payment delays:** coordinate with finance to understand disputes or invoice issues.
""")

if metrics:
    with st.expander("Model evaluation (held-out test set)"):
        st.caption("These metrics are calculated by train_model.py on a held-out test split.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        m2.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        m3.metric("F1 score", f"{metrics.get('f1', 0):.3f}")
        m4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        st.text("Confusion matrix [ [TN, FP], [FN, TP] ]")
        st.write(metrics.get("confusion_matrix"))
        st.text(metrics.get("classification_report", ""))

st.divider()
st.caption("Demo only: synthetic data and simulated churn labels. Validate with real, time-based customer/order data before any operational use.")
