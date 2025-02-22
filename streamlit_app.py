from matplotlib.pyplot import step
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF
from transformers import pipeline
from langdetect import detect
import math
import os
import re
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Set page configuration
st.set_page_config(
    page_title="AI Predictive Methods for Credit Underwriting",
    page_icon="💸",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ffffff, #e6f7ff);
            font-family: 'Arial', sans-serif;
        }
        .header-container {
            background: linear-gradient(to right, #4CAF50, #5ecf5e);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .header-container h1 {
            font-size: 40px;
        }
        .header-container p {
            font-size: 20px;
            margin-top: 5px;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="header-container">
        <h1>AI Predictive Methods for Credit Underwriting</h1>
        <p>Revolutionizing credit underwriting with AI-driven predictive analytics for smarter, faster decisions!</p>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import joblib
import pandas as pd
import re
from io import BytesIO
from fpdf import FPDF

# Load the trained ML model
model_path = 'best_features_model.pkl'
model = joblib.load(model_path)

# Initialize session state variables
if "loan_details" not in st.session_state:
    st.session_state["loan_details"] = {
        "full_name": "",
        "email": "",
        "phone": "",
        "cibil_score": 750,
        "income_annum": 5000000,
        "loan_amount": 2000000,
        "loan_term": 24,
        "loan_percent_income": 20.0,
        "active_loans": 1,
        "gender": "Men",
        "marital_status": "Single",
        "employee_status": "employed",
        "residence_type": "OWN",
        "loan_purpose": "Personal",
        "emi": None,
        "id_proof": None,
        "address_proof": None
    }

if "current_step" not in st.session_state:
    st.session_state["current_step"] = "Personal Information"

# Step Navigation Functions
def next_step():
    steps = ["Personal Information", "Loan Details", "Upload Documents", "Final Decision"]
    current_index = steps.index(st.session_state["current_step"])
    if current_index < len(steps) - 1:
        st.session_state["current_step"] = steps[current_index + 1]

def prev_step():
    steps = ["Personal Information", "Loan Details", "Upload Documents", "Final Decision"]
    current_index = steps.index(st.session_state["current_step"])
    if current_index > 0:
        st.session_state["current_step"] = steps[current_index - 1]

# Display the current step title
st.markdown(f"## {st.session_state['current_step']}")

# -------------------------
# STEP 1: PERSONAL INFORMATION
# -------------------------
if st.session_state["current_step"] == "Personal Information":
    full_name = st.text_input("Full Name", st.session_state["loan_details"]["full_name"])
    if full_name and not re.match(r"^[A-Za-z\s]+$", full_name):
        st.warning("⚠️ Please enter a valid name (only letters and spaces).")
    else:
        st.session_state["loan_details"]["full_name"] = full_name

    email = st.text_input("Email Address", st.session_state["loan_details"]["email"])
    if email and not re.match(r"^\S+@\S+\.\S+$", email):
        st.warning("⚠️ Please enter a valid email address.")
    else:
        st.session_state["loan_details"]["email"] = email

    phone = st.text_input("Phone Number", st.session_state["loan_details"]["phone"])
    if phone and not re.match(r"^\d{10}$", phone):
        st.warning("⚠️ Please enter a valid 10-digit phone number.")
    else:
        st.session_state["loan_details"]["phone"] = phone

    st.button("Next", on_click=next_step)

# -------------------------
# STEP 2: LOAN DETAILS
# -------------------------
elif st.session_state["current_step"] == "Loan Details":

    st.session_state["loan_details"]["cibil_score"] = st.slider("CIBIL Score (300-900):", 300, 900, st.session_state["loan_details"]["cibil_score"])
    st.session_state["loan_details"]["income_annum"] = st.number_input("Annual Income (INR):", min_value=0, step=10000, value=st.session_state["loan_details"]["income_annum"])
    st.session_state["loan_details"]["loan_amount"] = st.number_input("Loan Amount (INR):", min_value=0, step=10000, value=st.session_state["loan_details"]["loan_amount"])
    st.session_state["loan_details"]["loan_term"] = st.number_input("Loan Term (Months):", min_value=1, step=1, value=st.session_state["loan_details"]["loan_term"])
    st.session_state["loan_details"]["active_loans"] = st.number_input("Number of Active Loans:", min_value=0, step=1, value=st.session_state["loan_details"]["active_loans"])
    st.session_state["loan_details"]["gender"] = st.selectbox("Gender:", ["Men", "Women"], index=0 if st.session_state["loan_details"]["gender"] == "Men" else 1)
    st.session_state["loan_details"]["marital_status"] = st.selectbox("Marital Status:", ["Single", "Married"], index=0 if st.session_state["loan_details"]["marital_status"] == "Single" else 1)
    st.session_state["loan_details"]["employee_status"] = st.selectbox("Employment Status:", ["employed", "self employed", "unemployed", "student"], index=["employed", "self employed", "unemployed", "student"].index(st.session_state["loan_details"]["employee_status"]))
    st.session_state["loan_details"]["residence_type"] = st.selectbox("Residence Type:", ["MORTGAGE", "OWN", "RENT"], index=["MORTGAGE", "OWN", "RENT"].index(st.session_state["loan_details"]["residence_type"]))
    st.session_state["loan_details"]["loan_purpose"] = st.selectbox("Loan Purpose:", ["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"], index=["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"].index(st.session_state["loan_details"]["loan_purpose"]))
    
    
    # EMI Calculator
    st.markdown("### Loan EMI Calculator")
    loan_amount = st.session_state["loan_details"]["loan_amount"]
    loan_term_years = st.session_state["loan_details"]["loan_term"] / 12
    interest_rate = st.number_input("Interest Rate (%):", min_value=0.1, max_value=15.0, step=0.1, value=7.5)
    
    monthly_rate = interest_rate / (12 * 100)
    tenure_months = loan_term_years * 12
    
    if loan_amount > 0 and tenure_months > 0:
        emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
        st.session_state["loan_details"]["emi"] = emi
        st.write(f"**Estimated EMI:** ₹{emi:,.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Previous", on_click=prev_step)
    with col2:
        st.button("Next", on_click=next_step)

# -------------------------
# STEP 3: UPLOAD DOCUMENTS
# -------------------------
elif st.session_state["current_step"] == "Upload Documents":
    uploaded_id = st.file_uploader("Upload ID Proof")
    if uploaded_id is not None:
        st.session_state["loan_details"]["id_proof"] = uploaded_id

    uploaded_address = st.file_uploader("Upload Address Proof")
    if uploaded_address is not None:
        st.session_state["loan_details"]["address_proof"] = uploaded_address

    col1, col2 = st.columns(2)
    with col1:
        st.button("Previous", on_click=prev_step)
    with col2:
        st.button("Next", on_click=next_step)

# -------------------------
# STEP 4: FINAL DECISION
# -------------------------
elif st.session_state["current_step"] == "Final Decision":
    loan_details = st.session_state["loan_details"]

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "cibil_score": [loan_details["cibil_score"]],
        "income_annum": [loan_details["income_annum"]],
        "loan_amount": [loan_details["loan_amount"]],
        "loan_term": [loan_details["loan_term"]],
        "loan_percent_income": [loan_details["loan_percent_income"]],
        "active_loans": [loan_details["active_loans"]],
        "gender": [1 if loan_details["gender"] == "Women" else 0],
        "marital_status": [1 if loan_details["marital_status"] == "Married" else 0],
        "employee_status_self_employed": [1 if loan_details["employee_status"] == "self employed" else 0],
        "employee_status_unemployed": [1 if loan_details["employee_status"] == "unemployed" else 0],
        "employee_status_student": [1 if loan_details["employee_status"] == "student" else 0],
        "residence_type_OWN": [1 if loan_details["residence_type"] == "OWN" else 0],
        "residence_type_RENT": [1 if loan_details["residence_type"] == "RENT" else 0],
        "loan_purpose_Personal": [1 if loan_details["loan_purpose"] == "Personal" else 0],
        "loan_purpose_Home_Renovation": [1 if loan_details["loan_purpose"] == "Home Renovation" else 0],
        "loan_purpose_Education": [1 if loan_details["loan_purpose"] == "Education" else 0],
        "loan_purpose_Vehicle": [1 if loan_details["loan_purpose"] == "Vehicle" else 0],
    })
    
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    #prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.markdown("### Loan Rejected ❌")
            st.error(f"Rejection Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.markdown("### Loan Approved ✅")
            st.success(f"Approval Probability: {prediction_proba[0][0]:.2f}")

        # -----------------
        # PDF GENERATION
        # -----------------
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font('FreeSerif', '', 'FreeSerif.ttf', uni=True)
        pdf.set_font("FreeSerif", size=12)

        pdf.cell(200, 10, txt="Loan Approval Prediction Report", ln=True, align="C")
        pdf.ln(10)

        # Personal Information
        pdf.cell(200, 10, txt="Personal Information:", ln=True)
        pdf.cell(200, 10, txt=f"Full Name: {loan_details.get('full_name', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Email: {loan_details.get('email', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Phone: {loan_details.get('phone', 'N/A')}", ln=True)
        pdf.ln(10)

        # Loan Details
        pdf.cell(200, 10, txt="Loan Details:", ln=True)
        pdf.cell(200, 10, txt=f"CIBIL Score: {loan_details.get('cibil_score', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Loan Amount: ₹{loan_details.get('loan_amount', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Loan Term: {loan_details.get('loan_term', 'N/A')} months", ln=True)
        emi_value = loan_details.get("emi", None)
        if emi_value is not None:
            pdf.cell(200, 10, txt=f"Estimated EMI: ₹{emi_value:,.2f}", ln=True)
        else:
            pdf.cell(200, 10, txt="Estimated EMI: Not Calculated", ln=True)
        pdf.ln(10)

        # Prediction Results
        pdf.cell(200, 10, txt="Prediction Results:", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {'Approved' if prediction[0] == 0 else 'Rejected'}", ln=True)
        pdf.cell(200, 10, txt=f"Approval Probability: {prediction_proba[0][0]:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Rejection Probability: {prediction_proba[0][1]:.2f}", ln=True)
        pdf.ln(10)

        # Save PDF to buffer
        buffer = BytesIO()
        pdf.output(buffer, "S")
        buffer.seek(0)

        st.download_button(
        label="Download Report as PDF",
        data=buffer,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )
    # Function to generate PDF


# --- Sidebar Chatbot Header ---
st.sidebar.markdown("## 🤖 AI Financial Chatbot")

# --- Initialize Chat History & Session Variables ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "bot", "content": "👋 Hello! Ask about loans, EMI, credit scores, or investments!"}
    ]
if "last_topic" not in st.session_state:
    st.session_state["last_topic"] = None  
if "emi_active" not in st.session_state:
    st.session_state["emi_active"] = False  
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""  

# --- Chatbot Response System ---
def chatbot_response(user_message):
    user_message = user_message.lower().strip()
    
    responses = {
        "greetings": ["hello", "hi", "hey", "how are you"],
        "loan": ["loan", "borrow money", "finance", "lending"],
        "emi": ["emi", "monthly payment", "installment"],
        "credit_score": ["credit score", "cibil", "credit rating"]
    }
    
    loans = {
        "Personal Loan": "🏦 **Personal Loan:** ₹50K-₹25L | 10-15% Interest | No Collateral",
        "Business Loan": "💼 **Business Loan:** ₹5L-₹5Cr | 10-18% Interest | Requires Collateral",
        "Student Loan": "🎓 **Student Loan:** ₹1L-₹50L | 5-8% Interest",
        "Home Loan": "🏡 **Home Loan:** ₹10L-₹1Cr | 7-9% Interest",
        "Car Loan": "🚗 **Car Loan:** ₹1L-₹50L | 8-12% Interest"
    }
    
    if user_message in responses["greetings"]:
        return "👋 Hello! How can I assist you today?"
    
    if any(word in user_message for word in responses["loan"]):
        st.session_state["last_topic"] = "loan"
        return "📌 **Loan Help:** Select a loan type below."
    
    if st.session_state["last_topic"] == "loan":
        for key, response in loans.items():
            if key.lower() in user_message:
                st.session_state["last_topic"] = None
                return response
    
    if any(word in user_message for word in responses["emi"]):
        st.session_state["emi_active"] = True
        return "📊 **EMI Calculator Activated!** Enter loan details below."
    
    if any(word in user_message for word in responses["credit_score"]):
        return "🔍 **Credit Score Guide:**\n- **750+** = Excellent ✅\n- **650-749** = Good 👍\n- **550-649** = Fair ⚠️\n- **Below 550** = Poor ❌"
    
    return "🤖 Hmm, I don't have an answer for that. Try asking about loans, EMI, or investments!"

# --- Display Chat History ---
st.sidebar.markdown("### 💬 Chat History:")

for message in st.session_state["chat_messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(f"👤 **You:** {message['content']}")
    else:
        with st.chat_message("assistant"):
            st.write(f"🤖 **Bot:** {message['content']}")

# --- Loan Selection Dropdown ---
if st.session_state["last_topic"] == "loan":
    with st.sidebar:
        st.markdown("### 📌 Select a Loan Type:")
        
        loan_types = {
            "Personal Loan": "🏦 **Personal Loan:** ₹50K-₹25L | 10-15% Interest | No Collateral",
            "Business Loan": "💼 **Business Loan:** ₹5L-₹5Cr | 10-18% Interest | Requires Collateral",
            "Student Loan": "🎓 **Student Loan:** ₹1L-₹50L | 5-8% Interest",
            "Home Loan": "🏡 **Home Loan:** ₹10L-₹1Cr | 7-9% Interest",
            "Car Loan": "🚗 **Car Loan:** ₹1L-₹50L | 8-12% Interest"
        }

        selected_loan = st.selectbox("Select a Loan Type:", list(loan_types.keys()))

        if st.button("🔍 Get Loan Details"):
            # Add loan details as a bot message
            st.session_state["chat_messages"].append({"role": "bot", "content": loan_types[selected_loan]})
            st.session_state["last_topic"] = None  # Reset topic after selection
            st.rerun()


# --- User Input Field ---
user_input = st.sidebar.text_input("💬 Type your question:", value=st.session_state["user_input"], key="chat_input")

if st.sidebar.button("🚀 Send"):
    if user_input.strip():
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        bot_reply = chatbot_response(user_input)
        st.session_state["chat_messages"].append({"role": "bot", "content": bot_reply})
        st.session_state["user_input"] = ""
        st.rerun()

# --- EMI Calculator ---
if st.session_state["emi_active"]:
    st.sidebar.markdown("### 📊 EMI Calculator")
    loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=1000, value=500000, step=1000)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, value=10.0, step=0.1)
    tenure = st.sidebar.number_input("Tenure (Years)", min_value=1, value=5, step=1)
    
    if st.sidebar.button("📊 Calculate EMI"):
        r = (interest_rate / 12) / 100
        n = tenure * 12
        emi_result = round((loan_amount * r * (1 + r) ** n) / ((1 + r) ** n - 1), 2)
        st.sidebar.success(f"📌 Your Monthly EMI: ₹{emi_result:,}")

    if st.sidebar.button("🔄 Reset EMI Calculator"):
        st.session_state["emi_active"] = False
        st.rerun()

# --- Clear Chat History Button ---
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["chat_messages"] = []
    st.session_state["last_topic"] = None
    st.rerun()

# Footer
st.markdown(
    """
    <footer>
        <p>© 2025 AI Predictive Methods for Credit Underwriting. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
