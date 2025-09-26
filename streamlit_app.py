import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load trained model and feature columns
# -----------------------------
model = joblib.load("churn_model.pkl")
columns_used = joblib.load("feature_columns.pkl")

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")
st.write("""
This app predicts the probability that a customer will churn based on their account details and services.
""")

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Encode categorical inputs
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

# -----------------------------
# Create input DataFrame with all features
# -----------------------------
input_data = pd.DataFrame(0, index=[0], columns=columns_used)
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["Contract"] = contract_map[contract]
input_data["InternetService"] = internet_map[internet]

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Churn"):
    proba = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{proba:.2f}**")
    
    if proba > 0.5:
        st.error("⚠️ High Risk of Churn!")
    else:
        st.success("✅ Low Risk of Churn")

# -----------------------------
# Feature Importance Plot
# -----------------------------
st.subheader("🌟 Feature Importance")
try:
    importances = model.feature_importances_
    feature_names = columns_used
    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="viridis", ax=ax)
    ax.set_title("Top 10 Feature Importances")
    st.pyplot(fig)
except Exception as e:
    st.write("Could not generate feature importance plot:", e)

# -----------------------------
# Optional: Dataset Insights
# -----------------------------
st.subheader("Notes")
st.write("""
- Month-to-month contracts, high monthly charges, and short tenure are key drivers of churn.
- This app uses a Random Forest model trained on historical customer data.
- Use the sidebar to change inputs and see predictions in real-time.
""")

