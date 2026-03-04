import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="LoanSense AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern, 3D aesthetic
st.markdown("""
<style>
    /* Global background */
    .stApp {
        background: #f0f4f8;
    }
    
    /* 3D Container Styling */
    .st-emotion-cache-16txtl3 {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 20px 20px 60px #cdd1d5, -20px -20px 60px #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.4);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .st-emotion-cache-16txtl3:hover {
        transform: translateY(-5px);
    }
    
    /* Stylish 3D Button */
    .stButton>button {
        background: linear-gradient(145deg, #4f46e5, #423bdf);
        color: white;
        border-radius: 15px;
        padding: 0.8rem 2rem;
        font-weight: 800;
        font-size: 1.2rem;
        border: none;
        box-shadow: 5px 5px 10px #b1cbe3, -5px -5px 10px #ffffff;
        transition: all 0.2s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button:active {
        box-shadow: inset 5px 5px 10px #352fbb, inset -5px -5px 10px #695dff;
        transform: translateY(2px);
    }
    .stButton>button:hover {
        background: linear-gradient(145deg, #423bdf, #4f46e5);
    }

    /* 3D Cards for Results */
    .prediction-card-approved {
        background: linear-gradient(145deg, #12d191, #0fb07a);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 15px 15px 30px #cdd1d5, -15px -15px 30px #ffffff, inset 5px 5px 15px rgba(255,255,255,0.4);
        animation: popIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .prediction-card-rejected {
        background: linear-gradient(145deg, #ff4d4d, #e64545);
        color: white;
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        box-shadow: 15px 15px 30px #cdd1d5, -15px -15px 30px #ffffff, inset 5px 5px 15px rgba(255,255,255,0.3);
        animation: popIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    /* 3D Factor Items */
    .factor-item {
        margin: 15px 0;
        padding: 1.2rem;
        border-radius: 15px;
        background: #ffffff;
        color: #1f2937;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 5px 5px 10px #d9d9d9, -5px -5px 10px #ffffff;
        transition: transform 0.2s;
    }
    .factor-item:hover {
        transform: scale(1.02);
    }
    .factor-positive {
        border-left: 8px solid #10B981;
    }
    .factor-negative {
        border-left: 8px solid #EF4444;
    }
    
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.8) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    /* 3D Headings */
    h1 {
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.1);
    }
    h2, h3 {
        color: #1E3A8A;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Input field styling */
    .stNumberInput>div>div>input, .stSelectbox>div>div>div, .stSlider>div>div>div>div {
        border-radius: 10px;
        box-shadow: inset 3px 3px 6px #d9d9d9, inset -3px -3px 6px #ffffff;
        border: none !important;
        background: #f0f4f8;
    }
</style>
""", unsafe_allow_html=True)

import os

@st.cache_resource
def load_models():
    # If the user uploaded the files directly to the root directory on GitHub, look there instead
    model_dir = "models/" if os.path.exists("models/num_imputer.pkl") else ""
    
    num_imp = joblib.load(f'{model_dir}num_imputer.pkl')
    cat_imp = joblib.load(f'{model_dir}cat_imputer.pkl')
    le_edu = joblib.load(f'{model_dir}le_edu.pkl')
    ohe = joblib.load(f'{model_dir}ohe.pkl')
    scaler = joblib.load(f'{model_dir}scaler.pkl')
    model = joblib.load(f'{model_dir}log_model.pkl')
    ohe_cols = joblib.load(f'{model_dir}ohe_cols.pkl')
    ohe_feature_names = joblib.load(f'{model_dir}ohe_feature_names.pkl')
    return num_imp, cat_imp, le_edu, ohe, scaler, model, ohe_cols, ohe_feature_names

def main():
    st.markdown("<h1 style='text-align: center; margin-bottom: 0.5rem; font-size: 3.5rem;'>💸 LoanSense</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.2rem; margin-bottom: 3rem;'>AI-Powered Instant Loan Decisions & Financial Insights</p>", unsafe_allow_html=True)
    
    try:
        num_imp, cat_imp, le_edu, ohe, scaler, model, ohe_cols, ohe_feature_names = load_models()
    except Exception as e:
        st.error("⚠️ Model files not found. Please ensure all model artifacts exist in the 'models' directory.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 👤 Applicant Information")
        with st.container(border=True):
            app_income = st.number_input("Applicant Income ($)", min_value=0.0, value=10000.0, step=1000.0)
            coapp_income = st.number_input("Coapplicant Income ($)", min_value=0.0, value=0.0, step=1000.0)
            age = st.slider("Age", min_value=18, max_value=100, value=35)
            dependents = st.number_input("Number of Dependents", min_value=0.0, value=0.0, step=1.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            
    with col2:
        st.markdown("### 💼 Employment & Finances")
        with st.container(border=True):
            employ_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
            employ_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"])
            credit_score = st.slider("Credit Score", min_value=300.0, max_value=850.0, value=650.0, step=1.0)
            existing_loans = st.number_input("Number of Existing Loans", min_value=0.0, value=1.0, step=1.0)
            dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            savings = st.number_input("Total Savings ($)", min_value=0.0, value=5000.0, step=1000.0)

    st.markdown("### 📑 Loan Details")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000.0, value=20000.0, step=1000.0)
            loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"])
        with c2:
            loan_term = st.selectbox("Loan Term (Months)", [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 120.0, 180.0, 240.0, 360.0], index=3)
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        with c3:
            collateral_value = st.number_input("Collateral Value ($)", min_value=0.0, value=15000.0, step=1000.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Loan Status ✨"):
        with st.spinner("Processing Application..."):
            # Create DataFrame matching raw input format
            input_dict = {
                "Applicant_Income": [app_income],
                "Coapplicant_Income": [coapp_income],
                "Employment_Status": [employ_status],
                "Age": [age],
                "Marital_Status": [marital_status],
                "Dependents": [dependents],
                "Credit_Score": [credit_score],
                "Existing_Loans": [existing_loans],
                "DTI_Ratio": [dti_ratio],
                "Savings": [savings],
                "Collateral_Value": [collateral_value],
                "Loan_Amount": [loan_amount],
                "Loan_Term": [loan_term],
                "Loan_Purpose": [loan_purpose],
                "Property_Area": [property_area],
                "Education_Level": [education],
                "Gender": [gender],
                "Employer_Category": [employ_category]
            }
            
            X = pd.DataFrame(input_dict)
            
            # Since user inputs are complete, we technically don't need imputation,
            # but to be safe and align with the pipeline, we apply the loaded imputers
            numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
            
            X[numerical_cols] = num_imp.transform(X[numerical_cols])
            X[categorical_cols] = cat_imp.transform(X[categorical_cols])
            
            # Encode Education_Level
            X["Education_Level"] = le_edu.transform(X["Education_Level"])
            
            # One hot encoding
            encoded = ohe.transform(X[ohe_cols])
            encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=X.index)
            X = pd.concat([X.drop(columns=ohe_cols), encoded_df], axis=1)
            
            # Feature engineering
            X["DTI_Ratio_sq"] = X["DTI_Ratio"] ** 2
            X["Credit_Score_aq"] = X["Credit_Score"] ** 2
            X["Applicant_Income_log"] = np.log1p(X["Applicant_Income"])
            X = X.drop(columns=["DTI_Ratio", "Credit_Score"])
            
            # Ensure columns are in the exact order as training
            X = X[ohe_feature_names]
            
            # Scaling
            X_scaled = scaler.transform(X)
            
            # Prediction
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            # Calculate Contributions for Explainability
            contributions = model.coef_[0] * X_scaled[0]
            
            # Map clean names for display
            clean_names = {
                'Applicant_Income_log': 'Applicant Income',
                'Credit_Score_aq': 'Credit Score',
                'DTI_Ratio_sq': 'Debt-to-Income Ratio',
                'Loan_Amount': 'Loan Amount',
                'Loan_Term': 'Loan Term',
                'Age': 'Applicant Age',
                'Savings': 'Total Savings',
                'Collateral_Value': 'Collateral Value',
                'Existing_Loans': 'Existing Loans',
                'Dependents': 'Dependents'
            }
            
            factors = []
            for feat, cont in zip(ohe_feature_names, contributions):
                name = clean_names.get(feat, feat.replace('_', ' ').title())
                
                # Filter out Gender explicitly as per risk logic
                if "Gender" in name:
                    continue
                    
                # Skip zero contributions from one-hot encoding or tiny contributions
                if "Employment Status" in name or "Employer Category" in name or "Loan Purpose" in name or "Property Area" in name:
                    if abs(cont) < 0.01: continue 
                factors.append({"name": name, "impact": cont})
                
            factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
            top_factors = factors[:4] # Display top 4 impact factors
            
            st.markdown("<hr style='margin: 3rem 0; border: none; height: 1px; background: #E2E8F0;'>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                if prediction == 1:
                    prob = probability[1] * 100
                    st.markdown(f"""
                        <div class='prediction-card-approved'>
                            <h2 style='color: white; margin-bottom: 0.5rem;'>🎉 APPROVED</h2>
                            <h4 style='color: rgba(255,255,255,0.9); font-weight: 400;'>Your loan application was successful.</h4>
                            <div style='margin-top: 2rem; background: rgba(0,0,0,0.1); padding: 1rem; border-radius: 10px;'>
                                <p style='font-size: 1.1rem; margin: 0; opacity: 0.9;'>AI Confidence Score</p>
                                <p style='font-size: 2.5rem; font-weight: bold; margin: 0;'>{prob:.1f}%</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    prob = probability[0] * 100
                    st.markdown(f"""
                        <div class='prediction-card-rejected'>
                            <h2 style='color: white; margin-bottom: 0.5rem;'>⚠️ REJECTED</h2>
                            <h4 style='color: rgba(255,255,255,0.9); font-weight: 400;'>High risk detected. Policy declined.</h4>
                            <div style='margin-top: 2rem; background: rgba(0,0,0,0.1); padding: 1rem; border-radius: 10px;'>
                                <p style='font-size: 1.1rem; margin: 0; opacity: 0.9;'>AI Confidence Score</p>
                                <p style='font-size: 2.5rem; font-weight: bold; margin: 0;'>{prob:.1f}%</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown("### 🔍 Key Decision Drivers", unsafe_allow_html=True)
                st.markdown("<p style='color: #64748B;'>The AI's decision was most heavily influenced by these factors from your profile:</p>", unsafe_allow_html=True)
                
                for f in top_factors:
                    if f["impact"] > 0:
                        direction = "Positive Impact"
                        css_class = "factor-positive"
                        icon = "✅"
                    else:
                        direction = "Negative Risk"
                        css_class = "factor-negative"
                        icon = "⚠️" if prediction == 0 else "🔻"
                    
                    st.markdown(f"""
                        <div class='factor-item {css_class}'>
                            <span style='display: flex; align-items: center; gap: 8px;'>{icon} <b>{f['name']}</b></span>
                            <span style='font-size: 0.85rem; color: #6B7280;'>{direction}</span>
                        </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
