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
        .completed-section {
            background-color: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 5px solid #4CAF50;
        }
        .section-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .checkmark {
            color: #4CAF50;
            font-size: 24px;
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

# Load the trained model
model_path = 'best_features_model.pkl'
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Initialize session state
if "loan_details" not in st.session_state:
    st.session_state["loan_details"] = {
        "full_name": "",
        "email": "",
        "phone": "",
        "cibil_score": 750,
        "income_annum": 5000000,
        "loan_amount": 2000000,
        "loan_term": 24,
        "loan_percent_income": 0.0,  # Will be calculated
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

# Track completed sections
if "completed_sections" not in st.session_state:
    st.session_state["completed_sections"] = {
        "Personal Information": False,
        "Loan Details": False,
        "Upload Documents": False
    }

# Calculate loan percent income
def calculate_loan_percent():
    if st.session_state["loan_details"]["income_annum"] > 0:
        return (st.session_state["loan_details"]["loan_amount"] / st.session_state["loan_details"]["income_annum"]) * 100
    return 0.0

# Navigation steps
steps = ["Personal Information", "Loan Details", "Upload Documents", "Final Decision"]
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

# Progress bar
progress = st.progress(0)
progress_percentage = (st.session_state["current_step"] / (len(steps) - 1)) * 100
progress.progress(int(progress_percentage))

# Navigation buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.session_state["current_step"] > 0:
        if st.button("◀ Previous"):
            st.session_state["current_step"] -= 1
            st.rerun()
with col3:
    if st.session_state["current_step"] < len(steps) - 1:
        if st.button("Next ▶"):
            # Mark section as completed when moving to next
            current_section = steps[st.session_state["current_step"]]
            if current_section != "Final Decision":
                st.session_state["completed_sections"][current_section] = True
            st.session_state["current_step"] += 1
            st.rerun()
    else:
        if st.button("Submit Application"):
            st.success("Application submitted successfully!")
            st.balloons()

# Current step content
current_step = steps[st.session_state["current_step"]]

# Personal Information
if current_step == "Personal Information":
    st.markdown("### Step 1: Personal Information")
    
    # Check if section is completed
    if st.session_state["completed_sections"]["Personal Information"]:
        st.markdown('<div class="completed-section"><div class="section-title"><h3>Personal Information</h3><span class="checkmark">✓</span></div></div>', unsafe_allow_html=True)
    else:
        st.markdown("### Personal Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["loan_details"]["full_name"] = st.text_input("Full Name", st.session_state["loan_details"]["full_name"])
    with col2:
        st.session_state["loan_details"]["email"] = st.text_input("Email Address", st.session_state["loan_details"]["email"])
    
    st.session_state["loan_details"]["phone"] = st.text_input("Phone Number", st.session_state["loan_details"]["phone"])
    
    # Mark as complete button
    if st.button("✓ Mark Personal Information as Complete"):
        if st.session_state["loan_details"]["full_name"] and st.session_state["loan_details"]["email"] and st.session_state["loan_details"]["phone"]:
            st.session_state["completed_sections"]["Personal Information"] = True
            st.success("Personal Information marked as complete!")
            st.rerun()
        else:
            st.warning("Please fill all fields before marking as complete")

# Loan Details
elif current_step == "Loan Details":
    st.markdown("### Step 2: Loan Details")
    
    # Check if section is completed
    if st.session_state["completed_sections"]["Loan Details"]:
        st.markdown('<div class="completed-section"><div class="section-title"><h3>Loan Details</h3><span class="checkmark">✓</span></div></div>', unsafe_allow_html=True)
    else:
        st.markdown("### Loan Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["loan_details"]["cibil_score"] = st.slider("CIBIL Score (300-900):", 300, 900, st.session_state["loan_details"]["cibil_score"])
        st.session_state["loan_details"]["income_annum"] = st.number_input("Annual Income (INR):", min_value=0, step=10000, value=st.session_state["loan_details"]["income_annum"])
        st.session_state["loan_details"]["loan_amount"] = st.number_input("Loan Amount (INR):", min_value=0, step=10000, value=st.session_state["loan_details"]["loan_amount"])
        st.session_state["loan_details"]["loan_term"] = st.number_input("Loan Term (Months):", min_value=1, step=1, value=st.session_state["loan_details"]["loan_term"])
    
    with col2:
        # Calculate and display loan percent income (read-only)
        loan_percent = calculate_loan_percent()
        st.session_state["loan_details"]["loan_percent_income"] = loan_percent
        st.text_input("Loan Percent of Income (%):", value=f"{loan_percent:.2f}", disabled=True)
        
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
        st.write(f"**Estimated EMI:** Rs. {emi:,.2f}")
    else:
        st.session_state["loan_details"]["emi"] = None
        st.write("Please provide valid loan amount and term.")
    
    # Mark as complete button
    if st.button("✓ Mark Loan Details as Complete"):
        st.session_state["completed_sections"]["Loan Details"] = True
        st.success("Loan Details marked as complete!")
        st.rerun()

# Upload Documents
elif current_step == "Upload Documents":
    st.markdown("### Step 3: Upload Documents")
    
    # Check if section is completed
    if st.session_state["completed_sections"]["Upload Documents"]:
        st.markdown('<div class="completed-section"><div class="section-title"><h3>Upload Documents</h3><span class="checkmark">✓</span></div></div>', unsafe_allow_html=True)
    else:
        st.markdown("### Upload Documents")
    
    st.session_state["loan_details"]["id_proof"] = st.file_uploader("Upload ID Proof (Aadhaar, PAN, etc.)", type=["pdf", "jpg", "png"])
    st.session_state["loan_details"]["address_proof"] = st.file_uploader("Upload Address Proof (Utility bill, Rental agreement, etc.)", type=["pdf", "jpg", "png"])
    
    # Mark as complete button
    if st.button("✓ Mark Documents as Complete"):
        if st.session_state["loan_details"]["id_proof"] is not None and st.session_state["loan_details"]["address_proof"] is not None:
            st.session_state["completed_sections"]["Upload Documents"] = True
            st.success("Documents marked as complete!")
            st.rerun()
        else:
            st.warning("Please upload both documents before marking as complete")

# Final Decision
elif current_step == "Final Decision":
    st.markdown("### Step 4: Final Decision")
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

    # Prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.markdown("### Loan Rejected ❌")
            st.error(f"Rejection Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.markdown("### Loan Approved ✅")
            st.success(f"Approval Probability: {prediction_proba[0][0]:.2f}")

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", style="BU", size=12)
        pdf.cell(200, 10, txt="Loan Approval Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
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
        pdf.cell(200, 10, txt=f"Loan Amount: Rs. {loan_details.get('loan_amount', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Loan Term: {loan_details.get('loan_term', 'N/A')} months", ln=True)
        emi_value = loan_details.get("emi", None)
        if emi_value is not None:
            pdf.cell(200, 10, txt=f"Estimated EMI: Rs. {emi_value:,.2f}", ln=True)
        else:
            pdf.cell(200, 10, txt="Estimated EMI: Not Calculated", ln=True)
        pdf.ln(10)

        # Prediction Results
        pdf.cell(200, 10, txt="Prediction Results:", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {'Approved' if prediction[0] == 0 else 'Rejected'}", ln=True)
        pdf.cell(200, 10, txt=f"Approval Probability: {prediction_proba[0][0]:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Rejection Probability: {prediction_proba[0][1]:.2f}", ln=True)

        # Save PDF to buffer
        buffer = BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        buffer.write(pdf_bytes)
        buffer.seek(0)

        st.download_button(
            label="Download Report as PDF",
            data=buffer,
            file_name="loan_prediction_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown(
    """
    <footer>
        <p>© Harjit Singh Bhadauriya - 2025 AI Predictive Methods for Credit Underwriting. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)

# --- Chatbot in Sidebar ---
st.sidebar.markdown("## 🤖 AI Financial Chatbot with EMI Calculator")

# --- Initialize Chat History & Session Variables ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "bot", "content": "👋 Hello! You can speak or type your question.\n\n**📌 Categories:**\n- Loan Help 🏦\n- EMI Calculator 💰\n- Credit Score Info 🔍\n- Investments 📊\n- Business Loans 💼\n- Student Loans 🎓"}
    ]
if "last_topic" not in st.session_state:
    st.session_state["last_topic"] = None
if "emi_active" not in st.session_state:
    st.session_state["emi_active"] = False
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# --- Smarter Chatbot Response System ---
def chatbot_response(user_message):
    user_message = user_message.lower().strip()

    greetings = ["hello", "hi", "hey", "how are you"]
    if user_message in greetings:
        return "👋 Hello! How can I assist you today? You can ask about loans, EMI, or investments!"

    loan_topics = ["loan help", "loan", "finance", "borrow money"]
    if any(topic in user_message for topic in loan_topics):
        st.session_state["last_topic"] = "loan"
        return "📌 **Loan Help:**\n- **Personal Loans** 🏦\n- **Business Loans** 💼\n- **Student Loans** 🎓\n- **Home & Car Loans** 🚗🏡\n\n💡 Ask about a specific loan type for details!"

    loan_details = {
        "personal loan": """🏦 **Personal Loan Details:**
        - **Loan Amount:** ₹50,000 - ₹25 Lakh
        - **Interest Rate:** 10-15% per annum
        - **Collateral:** ❌ Not Required
        - **Repayment Tenure:** 1-5 years
        - **Processing Time:** ✅ 24-48 hours for approval""",
        
        "business loan": """💼 **Business Loan Guide:**
        - **Loan Amount:** ₹5 Lakh - ₹5 Crore
        - **Interest Rate:** 10-18% per annum
        - **Collateral:** ✅ Required for large loans
        - **Repayment Tenure:** 3-10 years""",
        
        "student loan": """🎓 **Student Loan Guide:**
        - **Loan Amount:** ₹1 Lakh - ₹50 Lakh
        - **Interest Rate:** 5-8% per annum
        - **Collateral:** ✅ Required for loans above ₹7.5 Lakh
        - **Repayment Tenure:** 10-15 years""",
        
        "home loan": """🏡 **Home Loan Details:**
        - **Loan Amount:** ₹10 Lakh - ₹1 Crore
        - **Interest Rate:** 7-9% per annum
        - **Collateral:** ✅ Property being purchased
        - **Repayment Tenure:** 10-30 years""",
        
        "car loan": """🚗 **Car Loan Details:**
        - **Loan Amount:** ₹1 Lakh - ₹50 Lakh
        - **Interest Rate:** 8-12% per annum
        - **Collateral:** ❌ Not Required (Car is the collateral)
        - **Repayment Tenure:** 1-7 years"""
    }

    for key, response in loan_details.items():
        if key in user_message:
            st.session_state["last_topic"] = key
            return response

    if st.session_state["last_topic"]:
        if "tell me more" in user_message or "more details" in user_message:
            if st.session_state["last_topic"] == "personal loan":
                return "🏦 **More on Personal Loans:**\n- Great for emergencies, vacations, or home improvements.\n- Processing time: **24-48 hours** in most banks."
            elif st.session_state["last_topic"] == "business loan":
                return "💼 **More on Business Loans:**\n- Best for expansion, working capital, and asset purchase.\n- Some banks offer **low-interest startup loans**."
            elif st.session_state["last_topic"] == "student loan":
                return "🎓 **More on Student Loans:**\n- Government banks offer **subsidized loans** for students from low-income families."
            elif st.session_state["last_topic"] == "home loan":
                return "🏡 **More on Home Loans:**\n- You can apply for **tax benefits** under Section 80C."
            elif st.session_state["last_topic"] == "car loan":
                return "🚗 **More on Car Loans:**\n- Special interest rates available for **electric vehicles (EVs)**."

    emi_keywords = ["emi", "monthly payment", "calculate emi"]
    if any(keyword in user_message for keyword in emi_keywords):
        st.session_state["emi_active"] = True
        return "📊 **EMI Calculator Activated!** Enter loan details below."

    if "credit score" in user_message or "cibil" in user_message:
        return "🔍 **Credit Score Guide:**\n- **750+** = Excellent ✅\n- **650-749** = Good 👍\n- **550-649** = Fair ⚠️\n- **Below 550** = Poor ❌\n\nHigher scores = Better loan rates!"

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
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        bot_reply = chatbot_response(user_input)
        st.session_state["chat_messages"].append({"role": "bot", "content": bot_reply})
        st.session_state["user_input"] = ""
        st.rerun()

# --- Display EMI Calculator if Triggered ---
if st.session_state["emi_active"]:
    loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=1000, value=500000, step=1000)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, value=10.0, step=0.1)
    tenure = st.sidebar.number_input("Tenure (Years)", min_value=1, value=5, step=1)
    
    if st.sidebar.button("📊 Calculate EMI"):
        emi_result = round((loan_amount * (interest_rate / 12 / 100) * (1 + (interest_rate / 12 / 100)) ** (tenure * 12)) / ((1 + (interest_rate / 12 / 100)) ** (tenure * 12) - 1), 2)
        st.sidebar.success(f"📌 Your Monthly EMI: ₹{emi_result:,}")
        st.session_state["emi_active"] = False
