import streamlit as st
import pickle
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="SmartLoan Predictor Portal", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))
le_dict = pickle.load(open("encoder.pkl","rb"))
feature_names = pickle.load(open("feature_names.pkl","rb"))

# ---------------- LOGIN ----------------
def login():
    st.title("🏦 Login Page")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- HEADER ----------------
st.title("🏦 AI Loan Risk Engine")
st.markdown("---")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender",["Male","Female"])
    married = st.selectbox("Married",["Yes","No"])
    dependents = st.selectbox("Dependents",["0","1","2","3+"])
    education = st.selectbox("Education",["Graduate","Not Graduate"])
    self_emp = st.selectbox("Self Employed",["Yes","No"])

with col2:
    app_income = st.number_input("Applicant Income (₹ Thousands)",0.0)
    co_income = st.number_input("Coapplicant Income (₹ Thousands)",0.0)
    loan_amt = st.number_input("Loan Amount (₹ Thousands)",0.0)
    term = st.number_input("Loan Term (months)",1)
    area = st.selectbox("Property Area",["Urban","Semiurban","Rural"])

st.markdown("---")

if st.button("🚀 Evaluate Loan"):

    app_id = "SB"+str(random.randint(100000,999999))

    if dependents=="3+":
        dependents=3
    else:
        dependents=int(dependents)

    total_income = app_income + co_income
    emi = loan_amt / term
    emi_rupees = round(emi * 1000)

    log_total_income = np.log(total_income + 1)
    log_loan_amt = np.log(loan_amt + 1)

    emi_to_income = emi / (total_income + 1)
    income_dep = total_income / (dependents + 1)

    high_income = 1 if total_income > 10 else 0
    short_term = 1 if term < 180 else 0
    large_loan = 1 if loan_amt > 250 else 0

    # Encode
    gender = le_dict['Gender'].transform([gender])[0]
    married = le_dict['Married'].transform([married])[0]
    education = le_dict['Education'].transform([education])[0]
    self_emp = le_dict['Self_Employed'].transform([self_emp])[0]
    area = le_dict['Property_Area'].transform([area])[0]

    user_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_emp,
        'ApplicantIncome': app_income,
        'CoapplicantIncome': co_income,
        'LoanAmount': loan_amt,
        'Loan_Amount_Term': term,
        'Property_Area': area,
        'TotalIncome': total_income,
        'LogTotalIncome': log_total_income,
        'LogLoanAmount': log_loan_amt,
        'EMI': emi,
        'EMI_to_Income': emi_to_income,
        'IncomePerDependent': income_dep,
        'HighIncomeFlag': high_income,
        'ShortTermFlag': short_term,
        'LargeLoanFlag': large_loan
    }

    user_df = pd.DataFrame([user_data])
    user_df = user_df[feature_names]

    prob = model.predict_proba(user_df)[0][1]

    # ---------------- DECISION ----------------
    st.subheader("📌 Loan Decision")

    if prob > 0.85:
        st.success("✅ Approved | Grade A")
    elif prob > 0.70:
        st.success("✅ Approved | Grade B")
    elif prob > 0.55:
        st.warning("⚠ Manual Review | Grade C")
    else:
        st.error("❌ Rejected | Grade D")

    st.subheader("📊 Approval Probability")
    st.progress(int(prob * 100))
    st.write(f"{prob:.2%}")

    # ---------------- EXPLANATION ----------------
    st.subheader("🧠 Decision Explanation")

    reasons = []

    if total_income < 6:
        reasons.append("Low total income compared to requested loan.")

    if emi_rupees > (total_income * 1000 * 0.4):
        reasons.append("EMI exceeds recommended 40% income threshold.")

    if loan_amt > 250:
        reasons.append("High loan amount increases risk.")

    if term < 120:
        reasons.append("Short loan tenure increases monthly burden.")

    if dependents >= 3:
        reasons.append("More dependents increase financial pressure.")

    if not reasons:
        reasons.append("Strong income profile and manageable EMI.")

    for r in reasons:
        st.write("•", r)

    # ---------------- FINANCIAL SUMMARY ----------------
    st.subheader("💼 Financial Summary")
    colA, colB = st.columns(2)
    colA.metric("Total Income (₹ Thousands)", f"{total_income:.2f}")
    colB.metric("EMI (Monthly)", f"₹{emi_rupees:,} per month")

    st.subheader("📄 Application ID")
    st.write(app_id)

st.markdown("---")
