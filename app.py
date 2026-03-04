import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, sleek aesthetic
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .prediction-card-approved {
        background: linear-gradient(135deg, #1f8a4c, #27ae60);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(39, 174, 96, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .prediction-card-rejected {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(231, 76, 60, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
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
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🏦 Smart Loan Approval Predictor</h1>", unsafe_allow_html=True)
    
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
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            if prediction == 1:
                prob = probability[1] * 100
                st.markdown(f"""
                    <div class='prediction-card-approved'>
                        <h2>🎉 Congratulations!</h2>
                        <h4>Your loan application is likely to be APPROVED.</h4>
                        <p style='font-size: 1.2rem; opacity: 0.9;'>Confidence Score: {prob:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                prob = probability[0] * 100
                st.markdown(f"""
                    <div class='prediction-card-rejected'>
                        <h2>⚠️ Application High Risk</h2>
                        <h4>Unfortunately, your loan application is likely to be REJECTED.</h4>
                        <p style='font-size: 1.2rem; opacity: 0.9;'>Confidence Score: {prob:.1f}%</p>
                        <p style='font-size: 0.9rem; margin-top: 1rem;'>Consider improving your credit score or reducing your debt-to-income ratio.</p>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
