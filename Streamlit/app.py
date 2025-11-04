# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
from pathlib import Path
import os
APP_DIR = Path(__file__).resolve().parent
os.chdir(APP_DIR)  # <- force working directory to the folder that has app.py + your model files


# Page config
st.set_page_config(
    page_title="FinSight - Smart Loan Approval",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .intro-container {
        text-align: center;
        padding: 100px 20px;
        animation: fadeIn 2s ease-in;
    }
    .intro-title {
        font-size: 72px;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        animation: zoomIn 1.5s ease-out;
    }
    .intro-subtitle {
        font-size: 24px;
        color: #666;
        margin-bottom: 30px;
        animation: fadeInUp 2s ease-out;
    }
    .progress-bar {
        height: 10px;
        background: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
        margin: 20px 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
    }
    .info-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    @keyframes zoomIn {
        from {
            transform: scale(0.5);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInUp {
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'show_intro' not in st.session_state:
    st.session_state.show_intro = True
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Introduction screen (Netflix-style)
if st.session_state.show_intro:
    st.markdown("""
        <div class="intro-container">
            <h1 class="intro-title">FinSight 💰</h1>
            <p class="intro-subtitle">AI-Powered Loan Approval Intelligence</p>
            <p style="font-size: 18px; color: #888; animation: fadeInUp 2.5s ease-out;">
                Built by <strong>Parth Patel</strong><br>
                TXST MSDAIS (Applied AI) Student<br>
                3+ Years Finance Experience at Major Canadian Bank
            </p>
        </div>
    """, unsafe_allow_html=True)

    time.sleep(3)
    st.session_state.show_intro = False
    st.rerun()


# Load all saved objects
@st.cache_resource
def load_models():
    model = joblib.load('FinSight_model.pkl')
    scaler = joblib.load('FinSight_scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    cols_to_scale = joblib.load('cols_to_scale.pkl')
    return model, scaler, feature_names, cols_to_scale


model, scaler, feature_names, cols_to_scale = load_models()


# Function to estimate interest rate based on credit score and loan type
def estimate_interest_rate(credit_score, product_type, loan_intent):
    """Estimate interest rate based on credit profile"""
    # Base rates by product type
    base_rates = {
        'Personal Loan': 8.0,
        'Line of Credit': 8.0,
        'Credit Card': 21.0
    }

    base_rate = base_rates.get(product_type, 8.0)

    # Adjust based on credit score
    if product_type == 'Credit Card':
        # Credit cards have smaller adjustments
        if credit_score >= 750:
            rate = base_rate - 3.0
        elif credit_score >= 700:
            rate = base_rate - 2.0
        elif credit_score >= 650:
            rate = base_rate
        elif credit_score >= 600:
            rate = base_rate + 2.0
        else:
            rate = base_rate + 4.0
    else:
        # Loans and Line of Credit
        if credit_score >= 750:
            rate = base_rate - 2.0
        elif credit_score >= 700:
            rate = base_rate - 1.0
        elif credit_score >= 650:
            rate = base_rate
        elif credit_score >= 600:
            rate = base_rate + 2.0
        else:
            rate = base_rate + 4.0

    # Adjust based on loan intent
    if loan_intent in ['Education', 'Home Improvement']:
        rate -= 0.5
    elif loan_intent in ['Debt Consolidation']:
        rate += 0.5

    return max(3.0, min(rate, 29.0))


# Progress bar
def show_progress(current_page, total_pages=5):
    progress = ((current_page) / total_pages) * 100
    st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress}%"></div>
        </div>
        <p style="text-align: center; color: #666;">Step {current_page} of {total_pages}</p>
    """, unsafe_allow_html=True)


# Header
st.title("💰 FinSight - Smart Loan Approval Predictor")

# Page 1: Personal Information
if st.session_state.page == 0:
    show_progress(1)
    st.markdown("### 👤 Personal Information")
    st.markdown("Let's start with some basic information about you.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100,
                              value=st.session_state.form_data.get('age', 35), step=1)
        occupation_status = st.selectbox("Occupation Status",
                                         ['Student', 'Self-Employed', 'Employed'],
                                         index=['Student', 'Self-Employed', 'Employed'].index(
                                             st.session_state.form_data.get('occupation_status', 'Employed')))

    with col2:
        # Dynamic help text based on occupation
        if occupation_status == 'Student':
            st.markdown(
                '<div class="info-box">💡 <strong>Tip:</strong> If you\'re a student with no work experience, enter 0 for years employed.</div>',
                unsafe_allow_html=True)
            default_years = 0.0
        else:
            default_years = st.session_state.form_data.get('years_employed', 5.0)

        years_employed = st.number_input("Years Employed",
                                         min_value=0.0,
                                         max_value=50.0,
                                         value=default_years,
                                         step=0.5,
                                         help="Enter 0 if you're a student or currently unemployed")

    st.markdown("---")
    col_next, col_empty = st.columns([1, 3])
    with col_next:
        if st.button("Next →", type="primary"):
            st.session_state.form_data['age'] = age
            st.session_state.form_data['occupation_status'] = occupation_status
            st.session_state.form_data['years_employed'] = years_employed
            st.session_state.page = 1
            st.rerun()

# Page 2: Financial Information
elif st.session_state.page == 1:
    show_progress(2)
    st.markdown("### 💵 Financial Information")
    st.markdown("Tell us about your financial situation.")

    col1, col2 = st.columns(2)

    with col1:
        annual_income = st.number_input("Annual Income ($)",
                                        min_value=0,
                                        max_value=1000000,
                                        value=st.session_state.form_data.get('annual_income', 50000),
                                        step=1000,
                                        help="Include all sources of income (salary, investments, etc.)")
        savings_assets = st.number_input("Savings & Assets ($)",
                                         min_value=0,
                                         max_value=1000000,
                                         value=st.session_state.form_data.get('savings_assets', 10000),
                                         step=1000,
                                         help="Total value of your savings accounts, investments, and assets")

    with col2:
        current_debt = st.number_input("Current Debt ($)",
                                       min_value=0,
                                       max_value=500000,
                                       value=st.session_state.form_data.get('current_debt', 5000),
                                       step=1000,
                                       help="Total outstanding debt (credit cards, car loans, mortgages, etc.)")

    st.markdown("---")
    col_back, col_next, col_empty = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = 0
            st.rerun()
    with col_next:
        if st.button("Next →", type="primary"):
            st.session_state.form_data['annual_income'] = annual_income
            st.session_state.form_data['savings_assets'] = savings_assets
            st.session_state.form_data['current_debt'] = current_debt
            st.session_state.page = 2
            st.rerun()

# Page 3: Credit Information
elif st.session_state.page == 2:
    show_progress(3)
    st.markdown("### 📊 Credit Information")
    st.markdown("Help us understand your credit history.")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=st.session_state.form_data.get('credit_score', 650),
                                       step=10,
                                       help="Check your credit score on Credit Karma, Experian, or your bank's app")
        credit_history_years = st.number_input("Credit History (Years)",
                                               min_value=0.0,
                                               max_value=50.0,
                                               value=st.session_state.form_data.get('credit_history_years', 5.0),
                                               step=0.5,
                                               help="How long have you had any form of credit?")

    with col2:
        defaults_on_file = st.selectbox("Any Defaults on File?",
                                        ['No', 'Yes'],
                                        index=['No', 'Yes'].index(
                                            st.session_state.form_data.get('defaults_on_file', 'No')),
                                        help="Have you ever defaulted on a loan or credit payment?")
        delinquencies_last_2yrs = st.number_input("Delinquencies (Last 2 Years)",
                                                  min_value=0,
                                                  max_value=20,
                                                  value=st.session_state.form_data.get('delinquencies_last_2yrs', 0),
                                                  step=1,
                                                  help="Number of times you were 30+ days late on payments")

    st.markdown("---")
    col_back, col_next, col_empty = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = 1
            st.rerun()
    with col_next:
        if st.button("Next →", type="primary"):
            st.session_state.form_data['credit_score'] = credit_score
            st.session_state.form_data['credit_history_years'] = credit_history_years
            st.session_state.form_data['defaults_on_file'] = defaults_on_file
            st.session_state.form_data['delinquencies_last_2yrs'] = delinquencies_last_2yrs
            st.session_state.page = 3
            st.rerun()

# Page 4: Loan Details
elif st.session_state.page == 3:
    show_progress(4)
    st.markdown("### 🏦 Loan Details")
    st.markdown("Finally, tell us about the loan you're applying for.")

    col1, col2 = st.columns(2)

    with col1:
        product_type = st.selectbox("Product Type",
                                    ['Line of Credit', 'Personal Loan', 'Credit Card'],
                                    index=['Line of Credit', 'Personal Loan', 'Credit Card'].index(
                                        st.session_state.form_data.get('product_type', 'Personal Loan')))
        loan_intent = st.selectbox("Purpose",
                                   ['Debt Consolidation', 'Business', 'Home Improvement',
                                    'Medical', 'Personal', 'Education'],
                                   index=['Debt Consolidation', 'Business', 'Home Improvement',
                                          'Medical', 'Personal', 'Education'].index(
                                       st.session_state.form_data.get('loan_intent', 'Personal')))
        loan_amount = st.number_input("Amount ($)",
                                      min_value=0,
                                      max_value=500000,
                                      value=st.session_state.form_data.get('loan_amount', 20000),
                                      step=1000)

    with col2:
        # Interest rate section with auto-estimation
        st.markdown(
            '<div class="info-box">💡 <strong>Don\'t know your interest rate?</strong><br>We\'ll estimate it based on your credit score and loan type. You can also enter a custom rate if you know it.</div>',
            unsafe_allow_html=True)

        use_estimated_rate = st.checkbox("Estimate interest rate for me",
                                         value=st.session_state.form_data.get('use_estimated_rate', True))

        if use_estimated_rate:
            credit_score_for_estimate = st.session_state.form_data.get('credit_score', 650)
            estimated_rate = estimate_interest_rate(credit_score_for_estimate, product_type, loan_intent)

            st.info(
                f"📊 **Estimated Interest Rate:** {estimated_rate:.2f}%\n\nBased on your credit score ({credit_score_for_estimate}) and loan type.")
            interest_rate = estimated_rate
        else:
            interest_rate = st.number_input("Interest Rate (%)",
                                            min_value=0.0,
                                            max_value=30.0,
                                            value=st.session_state.form_data.get('interest_rate', 5.0),
                                            step=0.1,
                                            help="The annual percentage rate (APR) for your loan")

    st.markdown("---")
    col_back, col_submit, col_empty = st.columns([1, 1, 2])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = 2
            st.rerun()
    with col_submit:
        if st.button("🔮 Get Prediction", type="primary"):
            st.session_state.form_data['product_type'] = product_type
            st.session_state.form_data['loan_intent'] = loan_intent
            st.session_state.form_data['loan_amount'] = loan_amount
            st.session_state.form_data['interest_rate'] = interest_rate
            st.session_state.form_data['use_estimated_rate'] = use_estimated_rate
            st.session_state.page = 4
            st.rerun()

# Page 5: Results
# Page 5: Results
elif st.session_state.page == 4:
    show_progress(5)

    with st.spinner("🔍 Analyzing your application..."):
        time.sleep(2)

        # Get all form data
        data = st.session_state.form_data

        # Calculate derived features
        # Handle zero income case - set ratios to very high value instead of 0
        if data['annual_income'] > 0:
            debt_to_income_ratio = data['current_debt'] / data['annual_income']
            loan_to_income_ratio = data['loan_amount'] / data['annual_income']
        else:
            debt_to_income_ratio = 999.0
            loan_to_income_ratio = 999.0

        # Estimate monthly payment
        if data['interest_rate'] > 0 and data['loan_amount'] > 0:
            monthly_rate = (data['interest_rate'] / 100) / 12
            num_payments = 60
            monthly_payment = data['loan_amount'] * (monthly_rate * (1 + monthly_rate) ** num_payments) / (
                        (1 + monthly_rate) ** num_payments - 1)

            if data['annual_income'] > 0:
                payment_to_income_ratio = (monthly_payment * 12) / data['annual_income']
            else:
                payment_to_income_ratio = 999.0
        else:
            payment_to_income_ratio = 0

        # Estimate derogatory marks
        if data['credit_score'] < 580:
            derogatory_marks = 3
        elif data['credit_score'] < 670:
            derogatory_marks = 1
        else:
            derogatory_marks = 0

        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [data['age']],
            'occupation_status': [data['occupation_status']],
            'years_employed': [data['years_employed']],
            'annual_income': [data['annual_income']],
            'credit_score': [data['credit_score']],
            'credit_history_years': [data['credit_history_years']],
            'defaults_on_file': [data['defaults_on_file']],
            'delinquencies_last_2yrs': [data['delinquencies_last_2yrs']],
            'derogatory_marks': [derogatory_marks],
            'savings_assets': [data['savings_assets']],
            'current_debt': [data['current_debt']],
            'debt_to_income_ratio': [debt_to_income_ratio],
            'loan_amount': [data['loan_amount']],
            'interest_rate': [data['interest_rate']],
            'product_type': [data['product_type']],
            'loan_intent': [data['loan_intent']],
            'loan_to_income_ratio': [loan_to_income_ratio],
            'payment_to_income_ratio': [payment_to_income_ratio]
        })

        # Apply encodings
        occupation_map = {'Student': 0, 'Self-Employed': 1, 'Employed': 2}
        input_data['occupation_status'] = input_data['occupation_status'].map(occupation_map)

        product_type_map = {'Line of Credit': 0, 'Personal Loan': 1, 'Credit Card': 3}
        input_data['product_type'] = input_data['product_type'].map(product_type_map)

        loan_intent_map = {
            'Debt Consolidation': 0,
            'Business': 1,
            'Home Improvement': 2,
            'Medical': 3,
            'Personal': 4,
            'Education': 5
        }
        input_data['loan_intent'] = input_data['loan_intent'].map(loan_intent_map)

        input_data['defaults_on_file'] = input_data['defaults_on_file'].map({'Yes': 1, 'No': 0})

        # Reorder columns
        input_data = input_data[feature_names]

        # Scale numerical columns
        input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

    # Display results
    st.markdown("## 📊 Prediction Results")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if prediction == 1:
            st.success("### ✅ APPROVED")
            st.balloons()
        else:
            st.error("### ❌ DECLINED")

    with col_b:
        st.metric("Approval Probability", f"{prediction_proba[1] * 100:.1f}%")

    with col_c:
        confidence = "High" if max(prediction_proba) > 0.8 else "Medium" if max(prediction_proba) > 0.6 else "Low"
        st.metric("Confidence Level", confidence)

    # Show advice for declined applications
    if prediction == 0:
        st.markdown("---")

        # Center the recommendations - remove columns, just use full width
        st.markdown("### 💡 Personalized Recommendations to Improve Your Approval Chances")

        recommendations = []

        # Special case: Zero income
        if data['annual_income'] == 0:
            recommendations.append({
                'icon': '💵',
                'title': 'Income Required',
                'desc': "You currently have no reported income. To qualify for most loans, you need verifiable income from employment, part-time work or consider adding a co-signer with income.",
                'priority': 'CRITICAL'
            })

        if data['credit_score'] < 670:
            recommendations.append({
                'icon': '🎯',
                'title': 'Improve Credit Score',
                'desc': f"Your credit score is {data['credit_score']}, which is below the preferred threshold of 670. Focus on: paying all bills on time, keeping credit utilization below 30%, and disputing any errors on your credit report.",
                'priority': 'HIGH'
            })

        # Only show DTI advice if income is not zero
        if data['annual_income'] > 0 and debt_to_income_ratio > 0.43:
            recommendations.append({
                'icon': '💰',
                'title': 'Reduce Debt-to-Income Ratio',
                'desc': f"Your DTI ratio is {debt_to_income_ratio:.1%}, which is above the recommended 43%. Pay down existing debts or increase your income to improve this ratio.",
                'priority': 'HIGH'
            })

        if data['delinquencies_last_2yrs'] > 0:
            recommendations.append({
                'icon': '📅',
                'title': 'Address Recent Delinquencies',
                'desc': f"You have {data['delinquencies_last_2yrs']} delinquenc{'y' if data['delinquencies_last_2yrs'] == 1 else 'ies'} in the last 2 years. Make all payments on time for the next 12-24 months to rebuild trust.",
                'priority': 'HIGH'
            })

        if data['defaults_on_file'] == 'Yes':
            recommendations.append({
                'icon': '⚠️',
                'title': 'Resolve Defaults',
                'desc': "Having defaults on file significantly impacts approval chances. Work with creditors to settle these accounts and have them marked as resolved.",
                'priority': 'CRITICAL'
            })

        # Only show loan-to-income advice if income is not zero
        if data['annual_income'] > 0 and loan_to_income_ratio > 0.5:
            recommendations.append({
                'icon': '📉',
                'title': 'Adjust Loan Amount',
                'desc': f"The requested amount (${data['loan_amount']:,}) is {loan_to_income_ratio:.1%} of your annual income. Consider requesting a smaller amount or wait until your income increases.",
                'priority': 'MEDIUM'
            })
        elif data['annual_income'] == 0 and data['loan_amount'] > 5000:
            recommendations.append({
                'icon': '📉',
                'title': 'Consider Smaller Amount',
                'desc': f"Without verifiable income, approval is very difficult for amounts over $5,000. For students, consider federal student loans, smaller credit card limits ($500-$2,000), or adding a co-signer.",
                'priority': 'HIGH'
            })

        if data['years_employed'] < 2 and data['occupation_status'] != 'Student':
            recommendations.append({
                'icon': '⏰',
                'title': 'Build Employment History',
                'desc': f"You've been employed for {data['years_employed']} years. Lenders typically prefer at least 2 years of stable employment history.",
                'priority': 'MEDIUM'
            })

        if data['savings_assets'] < data['loan_amount'] * 0.1:
            recommendations.append({
                'icon': '💵',
                'title': 'Increase Savings',
                'desc': f"Your savings (${data['savings_assets']:,}) are below 10% of the requested amount. Building an emergency fund demonstrates financial stability.",
                'priority': 'MEDIUM'
            })

        # Add specific advice for students with credit cards
        if data['occupation_status'] == 'Student' and data['product_type'] == 'Credit Card' and data[
            'annual_income'] == 0:
            recommendations.append({
                'icon': '💳',
                'title': 'Student Credit Card Options',
                'desc': "For student credit cards without income, consider: (1) Secured credit cards with a deposit, (2) Student credit cards that consider your enrollment status, or (3) Becoming an authorized user on a parent's card to build credit.",
                'priority': 'HIGH'
            })

        if not recommendations:
            recommendations.append({
                'icon': '📊',
                'title': 'Borderline Profile',
                'desc': "Your profile is close to approval threshold. Consider reapplying in 3-6 months after making minor improvements to your financial profile.",
                'priority': 'LOW'
            })

        # Display recommendations by priority - centered, full width
        for priority_level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
            priority_recs = [r for r in recommendations if r['priority'] == priority_level]
            if priority_recs:
                for rec in priority_recs:
                    if rec['priority'] == 'CRITICAL':
                        st.error(f"**{rec['icon']} {rec['title']}**\n\n{rec['desc']}")
                    elif rec['priority'] == 'HIGH':
                        st.warning(f"**{rec['icon']} {rec['title']}**\n\n{rec['desc']}")
                    else:
                        st.info(f"**{rec['icon']} {rec['title']}**\n\n{rec['desc']}")

        st.markdown("---")
        st.success(
            "💪 **General Tips**: Build credit history gradually, maintain low credit utilization (<30%), save for emergencies, and consider a co-signer if possible. Small improvements can make a big difference!")

    else:
        st.markdown("---")
        st.success(
            "🎉 **Congratulations!** Your application shows strong approval potential. Your financial profile demonstrates responsibility and stability. Proceed with confidence!")

        st.markdown("### ✨ Strengths of Your Application")
        strengths = []

        if data['credit_score'] >= 700:
            strengths.append("✅ Excellent credit score")
        if debt_to_income_ratio <= 0.36 and data['annual_income'] > 0:
            strengths.append("✅ Healthy debt-to-income ratio")
        if data['years_employed'] >= 3:
            strengths.append("✅ Strong employment history")
        if data['savings_assets'] >= data['loan_amount'] * 0.2:
            strengths.append("✅ Solid savings and assets")
        if data['delinquencies_last_2yrs'] == 0:
            strengths.append("✅ Clean recent payment history")

        for strength in strengths:
            st.markdown(strength)

    # Summary of application
    with st.expander("📋 Application Summary"):
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown("**Personal Information**")
            st.write(f"Age: {data['age']}")
            st.write(f"Occupation: {data['occupation_status']}")
            st.write(f"Years Employed: {data['years_employed']}")

            st.markdown("**Financial Information**")
            st.write(f"Annual Income: ${data['annual_income']:,}")
            st.write(f"Savings & Assets: ${data['savings_assets']:,}")
            st.write(f"Current Debt: ${data['current_debt']:,}")

        with summary_col2:
            st.markdown("**Credit Information**")
            st.write(f"Credit Score: {data['credit_score']}")
            st.write(f"Credit History: {data['credit_history_years']} years")
            st.write(f"Defaults on File: {data['defaults_on_file']}")
            st.write(f"Recent Delinquencies: {data['delinquencies_last_2yrs']}")

            st.markdown("**Loan Details**")
            st.write(f"Product Type: {data['product_type']}")
            st.write(f"Purpose: {data['loan_intent']}")
            st.write(f"Amount: ${data['loan_amount']:,}")
            st.write(f"Interest Rate: {data['interest_rate']:.2f}%")

    st.markdown("---")
    col_restart, col_empty = st.columns([1, 3])
    with col_restart:
        if st.button("🔄 New Application"):
            st.session_state.page = 0
            st.session_state.form_data = {}
            st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ About FinSight")
    st.info("""
    **FinSight** uses advanced machine learning to predict loan approval outcomes with **94% accuracy**.
    """)

    st.markdown("---")

    st.markdown("## 👨‍💻 About the Developer")
    st.success("""
    **Parth Patel**

    🎓 TXST MSDAIS Student  
    (Applied AI Specialization)

    💼 3+ Years Finance Experience  
    at Major Canadian Bank

    🚀 Passionate about AI in Finance
    """)

    st.markdown("---")

    st.markdown("## 📈 Model Performance")
    st.markdown("""
    - **Accuracy**: 94%
    - **Precision**: 94%
    - **Recall**: 94%
    - **F1-Score**: 94%
    """)

    st.markdown("---")

    st.markdown("## 🎯 Features")
    st.markdown("""
    ✅ Step-by-step process  
    ✅ Instant predictions  
    ✅ Personalized advice  
    ✅ Auto interest rate estimation  
    ✅ User-friendly interface
    """)
