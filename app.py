import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

# -----------------------------
# App Title & Description
# -----------------------------
st.title("🏦 Smart Loan Approval System")

st.write(
    """
    This system uses **Support Vector Machines (SVM)** to predict whether a loan
    should be **Approved or Rejected** based on applicant details.
    """
)

st.divider()

# -----------------------------
# Sidebar - User Inputs
# -----------------------------
st.sidebar.header("📋 Applicant Details")

applicant_income = st.sidebar.number_input(
    "Applicant Income", min_value=0, step=1000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount", min_value=0, step=1000
)

credit_history = st.sidebar.radio(
    "Credit History",
    ["Yes", "No"]
)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["Salaried", "Self-Employed", "Unemployed"]
)

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# -----------------------------
# Model Selection
# -----------------------------
st.sidebar.header("⚙️ Model Selection")

kernel_choice = st.sidebar.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# -----------------------------
# Encode Inputs
# -----------------------------
credit_encoded = 1 if credit_history == "Yes" else 0

employment_map = {
    "Salaried": 2,
    "Self-Employed": 1,
    "Unemployed": 0
}
employment_encoded = employment_map[employment_status]

property_map = {
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
}
property_encoded = property_map[property_area]

X_input = np.array([[
    applicant_income,
    loan_amount,
    credit_encoded,
    employment_encoded,
    property_encoded
]])

# -----------------------------
# Dummy Training Data (Demo)
# Replace with your trained model in real project
# -----------------------------
X_train = np.array([
    [50000, 100000, 1, 2, 2],
    [30000, 120000, 0, 1, 1],
    [70000, 150000, 1, 2, 2],
    [20000, 80000, 0, 0, 0],
    [90000, 200000, 1, 2, 1],
])

y_train = np.array([1, 0, 1, 0, 1])  # 1 = Approved, 0 = Rejected

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_input_scaled = scaler.transform(X_input)

# -----------------------------
# Model Selection
# -----------------------------
if kernel_choice == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel_choice == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", probability=True)

model.fit(X_train_scaled, y_train)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Check Loan Eligibility"):
    prediction = model.predict(X_input_scaled)[0]
    confidence = model.predict_proba(X_input_scaled).max() * 100

    st.divider()

    # -----------------------------
    # Output Section
    # -----------------------------
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    # -----------------------------
    # Business Explanation
    # -----------------------------
    st.subheader("📊 Business Explanation")

    if prediction == 1:
        st.write(
            """
            Based on the applicant's **income stability, credit history,
            and employment pattern**, the model predicts a **high likelihood
            of timely loan repayment**.
            """
        )
    else:
        st.write(
            """
            Based on the applicant's **credit history and income-risk pattern**,
            the model predicts a **higher risk of default**, hence the loan
            is rejected.
            """
        )
