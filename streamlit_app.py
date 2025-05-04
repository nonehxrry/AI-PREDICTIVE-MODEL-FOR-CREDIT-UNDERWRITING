#I Love My India
#Harjit
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF
from transformers import pipeline
from langdetect import detect
import math
import os

# Set page configuration with better theme
st.set_page_config(
    page_title="AI Credit Underwriting System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern styling
st.markdown(
    """
    <style>
        :root {
            --primary: #4f46e5;
            --secondary: #10b981;
            --accent: #f59e0b;
            --dark: #1e293b;
            --light: #f8fafc;
            --danger: #ef4444;
            --success: #10b981;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f9fafb;
            color: var(--dark);
        }
        
        .header-container {
            background: linear-gradient(135deg, var(--primary), #6366f1);
            color: white;
            padding: 2.5rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .header-container h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .header-container p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .progress-container {
            width: 100%;
            background-color: #e2e8f0;
            border-radius: 12px;
            margin-bottom: 2rem;
            height: 8px;
        }
        
        .progress-bar {
            height: 100%;
            border-radius: 12px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            transition: width 0.3s ease;
        }
        
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin-bottom: 2.5rem;
            position: relative;
        }
        
        .step-indicator::before {
            content: '';
            position: absolute;
            top: 15px;
            left: 0;
            right: 0;
            height: 2px;
            background-color: #e2e8f0;
            z-index: 1;
        }
        
        .step {
            text-align: center;
            width: 23%;
            position: relative;
            z-index: 2;
        }
        
        .step-number {
            width: 32px;
            height: 32px;
            background-color: #e2e8f0;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: #64748b;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .step-number.active {
            background-color: var(--primary);
            color: white;
            transform: scale(1.1);
        }
        
        .step-number.completed {
            background-color: var(--secondary);
            color: white;
        }
        
        .step-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: #64748b;
        }
        
        .step.active .step-label {
            color: var(--primary);
            font-weight: 600;
        }
        
        .step.completed .step-label {
            color: var(--secondary);
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .form-card {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stButton>button:first-child {
            background-color: white;
            color: var(--primary);
            border: 1px solid var(--primary);
        }
        
        .stButton>button:first-child:hover {
            background-color: #f5f3ff;
        }
        
        .stButton>button:nth-child(2) {
            background-color: var(--primary);
            color: white;
        }
        
        .stButton>button:nth-child(2):hover {
            background-color: #4338ca;
        }
        
        .stNumberInput>div>div>input, .stTextInput>div>div>input {
            background-color: #f8fafc !important;
            border-radius: 8px !important;
        }
        
        .stSelectbox>div>div>div {
            border-radius: 8px !important;
        }
        
        .file-uploader {
            border: 2px dashed #cbd5e1;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1rem;
            background-color: #f8fafc;
        }
        
        .file-uploader:hover {
            border-color: var(--primary);
        }
        
        .summary-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        .summary-card h3 {
            color: var(--primary);
            margin-bottom: 1rem;
        }
        
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin: 2rem 0;
        }
        
        .approved {
            border-left: 6px solid var(--success);
        }
        
        .rejected {
            border-left: 6px solid var(--danger);
        }
        
        .result-title {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .result-probability {
            font-size: 1.25rem;
            font-weight: 600;
        }
        
        .stDownloadButton>button {
            background-color: var(--success) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 500 !important;
            width: 100% !important;
        }
        
        .stDownloadButton>button:hover {
            background-color: #0d9b6c !important;
        }
        
        footer {
            text-align: center;
            margin-top: 3rem;
            font-size: 0.9rem;
            color: #64748b;
        }
        
        /* Chatbot specific styles */
        .chat-container {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .user-message {
            background-color: #f1f5f9;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            max-width: 80%;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        .bot-message {
            background-color: #e0e7ff;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            max-width: 80%;
            margin-right: auto;
            border-bottom-left-radius: 4px;
        }
        
        .chat-input {
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
        }
        
        .chat-button {
            width: 100% !important;
            border-radius: 12px !important;
            background-color: var(--primary) !important;
        }
        
        @media (max-width: 768px) {
            .header-container h1 {
                font-size: 1.8rem;
            }
            
            .header-container {
                padding: 1.5rem;
            }
            
            .form-card {
                padding: 1.5rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with improved design
st.markdown(
    """
    <div class="header-container">
        <h1>AI-Powered Credit Underwriting</h1>
        <p>Smart, fast, and fair loan decisions powered by machine learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model_path = 'best_features_model.pkl'
try:
    model = joblib.load(model_path)
    st.success("AI model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Initialize session state with better structure
if "loan_details" not in st.session_state:
    st.session_state["loan_details"] = {
        "personal_info": {
            "full_name": "",
            "email": "",
            "phone": "",
            "gender": "Men",
            "marital_status": "Single",
            "employee_status": "employed",
            "residence_type": "OWN"
        },
        "loan_details": {
            "cibil_score": 750,
            "income_annum": 500000,
            "loan_amount": 200000,
            "loan_term": 24,
            "loan_percent_income": 0.0,
            "active_loans": 1,
            "loan_purpose": "Personal",
            "emi": None,
            "interest_rate": 7.5
        },
        "documents": {
            "id_proof": None,
            "address_proof": None
        },
        "current_step": 1,
        "steps_completed": {
            "personal_info": False,
            "loan_details": False,
            "documents": False
        }
    }

# Calculate loan percent of income
def calculate_loan_percent():
    try:
        income = st.session_state["loan_details"]["loan_details"]["income_annum"]
        loan_amount = st.session_state["loan_details"]["loan_details"]["loan_amount"]
        if income > 0:
            return (loan_amount / income) * 100
        return 0.0
    except:
        return 0.0

# Update loan percent when income or loan amount changes
if "income_annum" in st.session_state["loan_details"]["loan_details"] and "loan_amount" in st.session_state["loan_details"]["loan_details"]:
    st.session_state["loan_details"]["loan_details"]["loan_percent_income"] = calculate_loan_percent()

# Progress bar with improved design
def show_progress():
    completed_steps = sum(st.session_state["loan_details"]["steps_completed"].values())
    total_steps = len(st.session_state["loan_details"]["steps_completed"])
    progress = (completed_steps / total_steps) * 100
    
    st.markdown(
        f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress}%"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    current_step = st.session_state["loan_details"]["current_step"]
    steps = [
        ("Personal Info", 1),
        ("Loan Details", 2),
        ("Documents", 3),
        ("Decision", 4)
    ]
    
    step_html = '<div class="step-indicator">'
    for label, step in steps:
        active = "active" if current_step == step else ""
        completed = "completed" if (
            (step == 1 and st.session_state["loan_details"]["steps_completed"]["personal_info"]) or
            (step == 2 and st.session_state["loan_details"]["steps_completed"]["loan_details"]) or
            (step == 3 and st.session_state["loan_details"]["steps_completed"]["documents"])
        ) else ""
        
        step_class = "step"
        if active:
            step_class += " active"
        if completed:
            step_class += " completed"
        
        step_html += f"""
        <div class="{step_class}">
            <div class="step-number {active} {completed}">{step}</div>
            <div class="step-label">{label}</div>
        </div>
        """
    step_html += '</div>'
    
    st.markdown(step_html, unsafe_allow_html=True)

# Navigation functions
def next_step():
    current = st.session_state["loan_details"]["current_step"]
    if current < 4:
        # Validate current step before proceeding
        if current == 1:
            personal_info = st.session_state["loan_details"]["personal_info"]
            if (personal_info["full_name"] and 
                personal_info["email"] and 
                personal_info["phone"]):
                st.session_state["loan_details"]["steps_completed"]["personal_info"] = True
                st.session_state["loan_details"]["current_step"] += 1
            else:
                st.error("Please complete all required personal information fields")
        elif current == 2:
            st.session_state["loan_details"]["steps_completed"]["loan_details"] = True
            st.session_state["loan_details"]["current_step"] += 1
        elif current == 3:
            st.session_state["loan_details"]["steps_completed"]["documents"] = True
            st.session_state["loan_details"]["current_step"] += 1

def prev_step():
    if st.session_state["loan_details"]["current_step"] > 1:
        st.session_state["loan_details"]["current_step"] -= 1

# Main application
show_progress()

# Step 1: Personal Information
if st.session_state["loan_details"]["current_step"] == 1:
    st.markdown('<div class="section-title">📝 Personal Information</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state["loan_details"]["personal_info"]["full_name"] = st.text_input(
                "Full Name*", 
                st.session_state["loan_details"]["personal_info"]["full_name"],
                placeholder="Enter your full name"
            )
            st.session_state["loan_details"]["personal_info"]["email"] = st.text_input(
                "Email Address*", 
                st.session_state["loan_details"]["personal_info"]["email"],
                placeholder="your.email@example.com"
            )
        
        with col2:
            st.session_state["loan_details"]["personal_info"]["phone"] = st.text_input(
                "Phone Number*", 
                st.session_state["loan_details"]["personal_info"]["phone"],
                placeholder="+91 9876543210"
            )
            st.session_state["loan_details"]["personal_info"]["gender"] = st.selectbox(
                "Gender:", 
                ["Men", "Women"], 
                index=0 if st.session_state["loan_details"]["personal_info"]["gender"] == "Men" else 1
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.session_state["loan_details"]["personal_info"]["marital_status"] = st.selectbox(
                "Marital Status:", 
                ["Single", "Married"], 
                index=0 if st.session_state["loan_details"]["personal_info"]["marital_status"] == "Single" else 1
            )
        
        with col4:
            st.session_state["loan_details"]["personal_info"]["employee_status"] = st.selectbox(
                "Employment Status:", 
                ["employed", "self employed", "unemployed", "student"], 
                index=["employed", "self employed", "unemployed", "student"].index(
                    st.session_state["loan_details"]["personal_info"]["employee_status"]
                )
            )
        
        st.session_state["loan_details"]["personal_info"]["residence_type"] = st.selectbox(
            "Residence Type:", 
            ["MORTGAGE", "OWN", "RENT"], 
            index=["MORTGAGE", "OWN", "RENT"].index(
                st.session_state["loan_details"]["personal_info"]["residence_type"]
            )
        )
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close form-card
    
    # Navigation buttons with better styling
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◄ Previous", disabled=True, key="prev_personal")
    with col2:
        st.button("Next ►", on_click=next_step, type="primary", key="next_personal")

# Step 2: Loan Details
elif st.session_state["loan_details"]["current_step"] == 2:
    st.markdown('<div class="section-title">💰 Loan Details</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state["loan_details"]["loan_details"]["cibil_score"] = st.slider(
                "CIBIL Score (300-900):", 
                300, 900, st.session_state["loan_details"]["loan_details"]["cibil_score"],
                help="Your credit score between 300-900"
            )
            
            st.session_state["loan_details"]["loan_details"]["income_annum"] = st.number_input(
                "Annual Income (INR):*", 
                min_value=0, step=10000, 
                value=st.session_state["loan_details"]["loan_details"]["income_annum"],
                key="income_input",
                format="%d"
            )
            
            st.session_state["loan_details"]["loan_details"]["loan_amount"] = st.number_input(
                "Loan Amount (INR):*", 
                min_value=0, step=10000, 
                value=st.session_state["loan_details"]["loan_details"]["loan_amount"],
                key="loan_amount_input",
                format="%d"
            )
            
            st.session_state["loan_details"]["loan_details"]["loan_term"] = st.number_input(
                "Loan Term (Months):*", 
                min_value=1, step=1, 
                value=st.session_state["loan_details"]["loan_details"]["loan_term"]
            )
            
            # Non-editable loan percent of income
            loan_percent = st.session_state["loan_details"]["loan_details"]["loan_percent_income"]
            st.number_input(
                "Loan Percent of Income (%):", 
                value=round(loan_percent, 2),
                disabled=True,
                key="loan_percent_disabled"
            )
        
        with col2:
            st.session_state["loan_details"]["loan_details"]["active_loans"] = st.number_input(
                "Number of Active Loans:", 
                min_value=0, step=1, 
                value=st.session_state["loan_details"]["loan_details"]["active_loans"]
            )
            
            st.session_state["loan_details"]["loan_details"]["loan_purpose"] = st.selectbox(
                "Loan Purpose:", 
                ["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"], 
                index=["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"].index(
                    st.session_state["loan_details"]["loan_details"]["loan_purpose"]
                )
            )
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close form-card
    
    # EMI Calculator in its own card
    with st.container():
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown("### 📊 EMI Calculator")
        
        emi_col1, emi_col2, emi_col3 = st.columns(3)
        
        with emi_col1:
            loan_amount = st.session_state["loan_details"]["loan_details"]["loan_amount"]
            st.number_input(
                "Loan Amount (INR)",
                value=loan_amount,
                disabled=True,
                key="emi_loan_amount"
            )
        
        with emi_col2:
            st.session_state["loan_details"]["loan_details"]["interest_rate"] = st.number_input(
                "Interest Rate (%):", 
                min_value=0.1, max_value=30.0, step=0.1, 
                value=st.session_state["loan_details"]["loan_details"]["interest_rate"]
            )
        
        with emi_col3:
            loan_term_years = st.session_state["loan_details"]["loan_details"]["loan_term"] / 12
            st.number_input(
                "Loan Term (Years)",
                value=round(loan_term_years, 1),
                disabled=True,
                key="emi_loan_term"
            )
        
        if st.button("Calculate EMI", key="calculate_emi"):
            try:
                monthly_rate = st.session_state["loan_details"]["loan_details"]["interest_rate"] / (12 * 100)
                tenure_months = st.session_state["loan_details"]["loan_details"]["loan_term"]
                loan_amount = st.session_state["loan_details"]["loan_details"]["loan_amount"]
                
                if loan_amount > 0 and tenure_months > 0:
                    emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / (
                        (1 + monthly_rate) ** tenure_months - 1
                    )
                    st.session_state["loan_details"]["loan_details"]["emi"] = emi
                    st.success(f"**Estimated Monthly EMI:** ₹{emi:,.2f}")
                else:
                    st.error("Please provide valid loan amount and term.")
            except Exception as e:
                st.error(f"Error calculating EMI: {str(e)}")
        
        if st.session_state["loan_details"]["loan_details"]["emi"]:
            st.info(f"Your calculated EMI is **₹{st.session_state['loan_details']['loan_details']['emi']:,.2f}** per month for {st.session_state['loan_details']['loan_details']['loan_term']} months.")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close form-card
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◄ Previous", on_click=prev_step, key="prev_loan")
    with col2:
        st.button("Next ►", on_click=next_step, type="primary", key="next_loan")

# Step 3: Upload Documents
elif st.session_state["loan_details"]["current_step"] == 3:
    st.markdown('<div class="section-title">📂 Upload Documents</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        
        st.info("Please upload clear scans of the required documents. Supported formats: PDF, JPG, PNG (Max 5MB each)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            st.session_state["loan_details"]["documents"]["id_proof"] = st.file_uploader(
                "Upload ID Proof (Aadhaar, PAN, Passport, etc.)", 
                type=["pdf", "jpg", "jpeg", "png"],
                key="id_proof_uploader"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            st.session_state["loan_details"]["documents"]["address_proof"] = st.file_uploader(
                "Upload Address Proof (Utility Bill, Rental Agreement, etc.)", 
                type=["pdf", "jpg", "jpeg", "png"],
                key="address_proof_uploader"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close form-card
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◄ Previous", on_click=prev_step, key="prev_docs")
    with col2:
        st.button("Next ►", on_click=next_step, type="primary", key="next_docs")

# Step 4: Final Decision
elif st.session_state["loan_details"]["current_step"] == 4:
    st.markdown('<div class="section-title">🎯 Application Review</div>', unsafe_allow_html=True)
    
    # Display summary in cards
    st.markdown("### 📝 Application Summary")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.markdown("#### Personal Information")
            st.write(f"**Name:** {st.session_state['loan_details']['personal_info']['full_name']}")
            st.write(f"**Email:** {st.session_state['loan_details']['personal_info']['email']}")
            st.write(f"**Phone:** {st.session_state['loan_details']['personal_info']['phone']}")
            st.write(f"**Employment:** {st.session_state['loan_details']['personal_info']['employee_status'].title()}")
            st.write(f"**Residence:** {st.session_state['loan_details']['personal_info']['residence_type']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.markdown("#### Loan Details")
            st.write(f"**Loan Amount:** ₹{st.session_state['loan_details']['loan_details']['loan_amount']:,}")
            st.write(f"**Loan Term:** {st.session_state['loan_details']['loan_details']['loan_term']} months")
            st.write(f"**CIBIL Score:** {st.session_state['loan_details']['loan_details']['cibil_score']}")
            st.write(f"**Annual Income:** ₹{st.session_state['loan_details']['loan_details']['income_annum']:,}")
            st.write(f"**Loan Purpose:** {st.session_state['loan_details']['loan_details']['loan_purpose']}")
            
            if st.session_state["loan_details"]["loan_details"]["emi"]:
                st.write(f"**Calculated EMI:** ₹{st.session_state['loan_details']['loan_details']['emi']:,.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction and results
    st.markdown("### 🔍 AI-Powered Decision")
    
    try:
        # Prepare input data for prediction
        personal_info = st.session_state["loan_details"]["personal_info"]
        loan_details = st.session_state["loan_details"]["loan_details"]
        
        input_data = pd.DataFrame({
            "cibil_score": [loan_details["cibil_score"]],
            "income_annum": [loan_details["income_annum"]],
            "loan_amount": [loan_details["loan_amount"]],
            "loan_term": [loan_details["loan_term"]],
            "loan_percent_income": [loan_details["loan_percent_income"]],
            "active_loans": [loan_details["active_loans"]],
            "gender": [1 if personal_info["gender"] == "Women" else 0],
            "marital_status": [1 if personal_info["marital_status"] == "Married" else 0],
            "employee_status_self_employed": [1 if personal_info["employee_status"] == "self employed" else 0],
            "employee_status_unemployed": [1 if personal_info["employee_status"] == "unemployed" else 0],
            "employee_status_student": [1 if personal_info["employee_status"] == "student" else 0],
            "residence_type_OWN": [1 if personal_info["residence_type"] == "OWN" else 0],
            "residence_type_RENT": [1 if personal_info["residence_type"] == "RENT" else 0],
            "loan_purpose_Personal": [1 if loan_details["loan_purpose"] == "Personal" else 0],
            "loan_purpose_Home_Renovation": [1 if loan_details["loan_purpose"] == "Home Renovation" else 0],
            "loan_purpose_Education": [1 if loan_details["loan_purpose"] == "Education" else 0],
            "loan_purpose_Vehicle": [1 if loan_details["loan_purpose"] == "Vehicle" else 0],
        })
        
        # Ensure all required features are present
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Display result in a styled card
        result_class = "approved" if prediction[0] == 0 else "rejected"
        result_icon = "✅" if prediction[0] == 0 else "❌"
        result_text = "Approved" if prediction[0] == 0 else "Rejected"
        
        st.markdown(f'<div class="result-card {result_class}">', unsafe_allow_html=True)
        st.markdown(f'<div class="result-title">{result_icon} Loan {result_text}</div>', unsafe_allow_html=True)
        
        if prediction[0] == 0:
            st.success(f"**Approval Probability:** {prediction_proba[0][0]:.2%}")
            st.balloons()
        else:
            st.error(f"**Rejection Probability:** {prediction_proba[0][1]:.2%}")
            st.markdown("💡 *Consider improving your credit score or reducing your loan amount to increase approval chances.*")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Generate PDF Report
        def generate_pdf_report():
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Title
                pdf.set_font("Arial", style="B", size=16)
                pdf.cell(200, 10, txt="Loan Approval Prediction Report", ln=True, align="C")
                pdf.ln(10)
                
                # Personal Information
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 10, txt="Personal Information:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Full Name: {personal_info.get('full_name', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Email: {personal_info.get('email', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Phone: {personal_info.get('phone', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Gender: {personal_info.get('gender', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Marital Status: {personal_info.get('marital_status', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Employment Status: {personal_info.get('employee_status', 'N/A')}", ln=True)
                pdf.ln(5)
                
                # Loan Details
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 10, txt="Loan Details:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"CIBIL Score: {loan_details.get('cibil_score', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Annual Income: Rs. {loan_details.get('income_annum', 'N/A'):,}", ln=True)
                pdf.cell(200, 10, txt=f"Loan Amount: Rs. {loan_details.get('loan_amount', 'N/A'):,}", ln=True)
                pdf.cell(200, 10, txt=f"Loan Term: {loan_details.get('loan_term', 'N/A')} months", ln=True)
                pdf.cell(200, 10, txt=f"Loan Purpose: {loan_details.get('loan_purpose', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Loan Percent of Income: {loan_details.get('loan_percent_income', 'N/A'):.2f}%", ln=True)
                
                emi_value = loan_details.get("emi", None)
                if emi_value is not None:
                    pdf.cell(200, 10, txt=f"Estimated EMI: Rs. {emi_value:,.2f}", ln=True)
                else:
                    pdf.cell(200, 10, txt="Estimated EMI: Not Calculated", ln=True)
                pdf.ln(5)
                
                # Prediction Results
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 10, txt="Prediction Results:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Prediction: {'Approved' if prediction[0] == 0 else 'Rejected'}", ln=True)
                pdf.cell(200, 10, txt=f"Approval Probability: {prediction_proba[0][0]:.2%}", ln=True)
                pdf.cell(200, 10, txt=f"Rejection Probability: {prediction_proba[0][1]:.2%}", ln=True)
                
                # Save PDF to buffer
                buffer = BytesIO()
                pdf_bytes = pdf.output(dest="S").encode("latin1")
                buffer.write(pdf_bytes)
                buffer.seek(0)
                return buffer
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                return None
        
        # Download button for PDF report
        pdf_buffer = generate_pdf_report()
        if pdf_buffer:
            st.download_button(
                label="📄 Download Full Report (PDF)",
                data=pdf_buffer,
                file_name="loan_decision_report.pdf",
                mime="application/pdf"
            )
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("◄ Previous", on_click=prev_step, key="prev_decision")
    with col2:
        if st.button("Submit Application", type="primary", key="submit_application"):
            st.success("🎉 Application submitted successfully!")
            # Here you would typically send the data to your backend
            # For demo purposes, we'll just show a success message

# Footer with improved design
st.markdown(
    """
    <footer>
        <p>© 2025 AI Credit Underwriting System | Developed with ❤️ in India</p>
    </footer>
    """,
    unsafe_allow_html=True
)

# --- Enhanced Chatbot in Sidebar ---
st.sidebar.markdown("## 🤖 Financial Assistant")

# Initialize Chat History
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "bot", "content": "👋 Hello! I'm your financial assistant. How can I help you today?\n\nYou can ask about:\n- **Loan options** 🏦\n- **EMI calculations** 💰\n- **Credit score tips** 🔍\n- **Application process** 📝"}
    ]

# --- Smarter Chatbot Response System ---
def chatbot_response(user_message):
    user_message = user_message.lower().strip()

    # Standard Greetings
    greetings = ["hello", "hi", "hey", "how are you"]
    if user_message in greetings:
        return "👋 Hello! How can I assist you today? You can ask about loans, EMI, or investments!"

    # Loan Categories
    loan_topics = ["loan help", "loan", "finance", "borrow money"]
    if any(topic in user_message for topic in loan_topics):
        st.session_state["last_topic"] = "loan"
        return "📌 **Loan Help:**\n- **Personal Loans** 🏦\n- **Business Loans** 💼\n- **Student Loans** 🎓\n- **Home & Car Loans** 🚗🏡\n\n💡 Ask about a specific loan type for details!"

    # Specific Loans with More Details
    loan_details = {
        "personal loan": """🏦 **Personal Loan Details:**
        - **Loan Amount:** ₹50,000 - ₹25 Lakh
        - **Interest Rate:** 10-15% per annum
        - **Collateral:** ❌ Not Required
        - **Repayment Tenure:** 1-5 years
        - **Processing Time:** ✅ 24-48 hours for approval
        - **Eligibility:**
        - CIBIL Score: **700+**
        - Monthly Income: **₹25,000+**
        - Age: **21-60 years**
        - **Best for:** Medical emergencies, vacations, home renovations, and debt consolidation.""",
        
        "business loan": """💼 **Business Loan Guide:**
        - **Loan Amount:** ₹5 Lakh - ₹5 Crore (Varies by bank)
        - **Interest Rate:** 10-18% per annum
        - **Collateral:** ✅ Required for large loans (property, assets)
        - **Repayment Tenure:** 3-10 years
        - **Processing Time:** 📅 7-15 days
        - **Eligibility:**
        - Business Age: **2+ years**
        - Annual Revenue: **₹10 Lakh+**
        - Good credit history
        - **Best for:** Expanding operations, working capital, asset purchase.""",
        
        "student loan": """🎓 **Student Loan Guide:**
        - **Loan Amount:** ₹1 Lakh - ₹50 Lakh
        - **Interest Rate:** 5-8% per annum (Lower for government schemes)
        - **Collateral:** ✅ Required for loans above ₹7.5 Lakh
        - **Repayment Tenure:** 10-15 years (Starts after graduation)
        - **Processing Time:** 📅 5-10 days
        - **Eligibility:**
        - Must be admitted to a recognized institution
        - Co-applicant (Parent/Guardian) with stable income
        - CIBIL Score: **650+**
        - **Best for:** Tuition, living expenses, and study abroad costs.""",
    }

    # Check for a specific loan type
    for key, response in loan_details.items():
        if key in user_message:
            st.session_state["last_topic"] = key  # Store last topic
            return response

    # EMI Calculator Activation
    emi_keywords = ["emi", "monthly payment", "calculate emi"]
    if any(keyword in user_message for keyword in emi_keywords):
        st.session_state["emi_active"] = True
        return "📊 **EMI Calculator Activated!** Enter loan details below."

    # Credit Score
    if "credit score" in user_message or "cibil" in user_message:
        return "🔍 **Credit Score Guide:**\n- **750+** = Excellent ✅\n- **650-749** = Good 👍\n- **550-649** = Fair ⚠️\n- **Below 550** = Poor ❌\n\nHigher scores = Better loan rates!"

    # Default Response
    return "🤖 Hmm, I don't have an exact answer for that. Try asking about loans, EMI, or investments!"

# --- Display Chat History ---
st.sidebar.markdown("### 💬 Chat History:")
for message in st.session_state["chat_messages"]:
    role = "👤 You" if message["role"] == "user" else "🤖 Bot"
    st.sidebar.markdown(f"**{role}:** {message['content']}")

# --- Text Input Field for Manual Chat ---
user_input = st.sidebar.text_input("💬 Type your question:", value=st.session_state["user_input"], key="chat_input")

# --- Process User Input ---
if st.sidebar.button("🚀 Send"):
    if user_input.strip():
        # Add user input to chat history
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        
        # Get bot response
        bot_reply = chatbot_response(user_input)
        st.session_state["chat_messages"].append({"role": "bot", "content": bot_reply})
        
        # Clear input field by resetting session state
        st.session_state["user_input"] = ""  
        
        # Refresh UI to show cleared input field
        st.rerun()

# --- Display EMI Calculator if Triggered ---
if st.session_state["emi_active"]:
    st.sidebar.markdown("### 📊 EMI Calculator")
    loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=1000, value=500000, step=1000)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, value=10.0, step=0.1)
    tenure = st.sidebar.number_input("Tenure (Years)", min_value=1, value=5, step=1)
    
    if st.sidebar.button("Calculate EMI"):
        try:
            monthly_rate = interest_rate / (12 * 100)
            emi_result = (loan_amount * monthly_rate * (1 + monthly_rate) ** (tenure * 12)) / (
                (1 + monthly_rate) ** (tenure * 12) - 1
            )
            st.sidebar.success(f"📌 Your Monthly EMI: ₹{emi_result:,.2f}")
            st.session_state["emi_active"] = False  # Reset EMI trigger
        except Exception as e:
            st.sidebar.error(f"Error calculating EMI: {str(e)}")
