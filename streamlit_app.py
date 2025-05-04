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
        .progress-container {
            width: 100%;
            background-color: #f3f3f3;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .progress-bar {
            height: 20px;
            border-radius: 10px;
            background-color: #4CAF50;
            text-align: center;
            line-height: 20px;
            color: white;
        }
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .step {
            text-align: center;
            width: 23%;
        }
        .step-number {
            width: 30px;
            height: 30px;
            background-color: #ddd;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 5px;
        }
        .step-number.active {
            background-color: #4CAF50;
            color: white;
        }
        .step-number.completed {
            background-color: #2E7D32;
            color: white;
        }
        .step-label {
            font-size: 12px;
        }
        .navigation-buttons {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #666;
        }
        .stNumberInput > div > div > input {
            background-color: #f5f5f5 !important;
            cursor: not-allowed !important;
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
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Initialize session state
if "loan_details" not in st.session_state:
    st.session_state["loan_details"] = {
        "full_name": "",
        "email": "",
        "phone": "",
        "cibil_score": 750,
        "income_annum": 500000,
        "loan_amount": 200000,
        "loan_term": 24,
        "loan_percent_income": 0.0,  # Will be calculated automatically
        "active_loans": 1,
        "gender": "Men",
        "marital_status": "Single",
        "employee_status": "employed",
        "residence_type": "OWN",
        "loan_purpose": "Personal",
        "emi": None,
        "id_proof": None,
        "address_proof": None,
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
        income = st.session_state["loan_details"]["income_annum"]
        loan_amount = st.session_state["loan_details"]["loan_amount"]
        if income > 0:
            return (loan_amount / income) * 100
        return 0.0
    except:
        return 0.0

# Update loan percent when income or loan amount changes
if "income_annum" in st.session_state["loan_details"] and "loan_amount" in st.session_state["loan_details"]:
    st.session_state["loan_details"]["loan_percent_income"] = calculate_loan_percent()

# Progress bar
def show_progress():
    completed_steps = sum(st.session_state["loan_details"]["steps_completed"].values())
    total_steps = len(st.session_state["loan_details"]["steps_completed"])
    progress = (completed_steps / total_steps) * 100
    
    st.markdown(
        f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress}%">{int(progress)}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    current_step = st.session_state["loan_details"]["current_step"]
    steps = [
        ("Personal Information", 1),
        ("Loan Details", 2),
        ("Upload Documents", 3),
        ("Final Decision", 4)
    ]
    
    step_html = '<div class="step-indicator">'
    for label, step in steps:
        active = "active" if current_step == step else ""
        completed = "completed" if (
            (step == 1 and st.session_state["loan_details"]["steps_completed"]["personal_info"]) or
            (step == 2 and st.session_state["loan_details"]["steps_completed"]["loan_details"]) or
            (step == 3 and st.session_state["loan_details"]["steps_completed"]["documents"])
        ) else ""
        
        step_html += f"""
        <div class="step">
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
            if (st.session_state["loan_details"]["full_name"] and 
                st.session_state["loan_details"]["email"] and 
                st.session_state["loan_details"]["phone"]):
                st.session_state["loan_details"]["steps_completed"]["personal_info"] = True
                st.session_state["loan_details"]["current_step"] += 1
            else:
                st.error("Please complete all personal information fields")
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
    st.markdown("### Step 1: Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state["loan_details"]["full_name"] = st.text_input(
            "Full Name*", 
            st.session_state["loan_details"]["full_name"]
        )
        st.session_state["loan_details"]["email"] = st.text_input(
            "Email Address*", 
            st.session_state["loan_details"]["email"]
        )
    
    with col2:
        st.session_state["loan_details"]["phone"] = st.text_input(
            "Phone Number*", 
            st.session_state["loan_details"]["phone"]
        )
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Previous", disabled=True)
    with col2:
        st.button("Next", on_click=next_step)

# Step 2: Loan Details
elif st.session_state["loan_details"]["current_step"] == 2:
    st.markdown("### Step 2: Loan Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state["loan_details"]["cibil_score"] = st.slider(
            "CIBIL Score (300-900):", 
            300, 900, st.session_state["loan_details"]["cibil_score"]
        )
        st.session_state["loan_details"]["income_annum"] = st.number_input(
            "Annual Income (INR):*", 
            min_value=0, step=10000, 
            value=st.session_state["loan_details"]["income_annum"],
            key="income_input"
        )
        st.session_state["loan_details"]["loan_amount"] = st.number_input(
            "Loan Amount (INR):*", 
            min_value=0, step=10000, 
            value=st.session_state["loan_details"]["loan_amount"],
            key="loan_amount_input"
        )
        st.session_state["loan_details"]["loan_term"] = st.number_input(
            "Loan Term (Months):*", 
            min_value=1, step=1, 
            value=st.session_state["loan_details"]["loan_term"]
        )
        
        # Non-editable loan percent of income
        loan_percent = st.session_state["loan_details"]["loan_percent_income"]
        st.number_input(
            "Loan Percent of Income (%):", 
            value=round(loan_percent, 2),
            disabled=True,
            key="loan_percent_disabled"
        )
    
    with col2:
        st.session_state["loan_details"]["active_loans"] = st.number_input(
            "Number of Active Loans:", 
            min_value=0, step=1, 
            value=st.session_state["loan_details"]["active_loans"]
        )
        st.session_state["loan_details"]["gender"] = st.selectbox(
            "Gender:", 
            ["Men", "Women"], 
            index=0 if st.session_state["loan_details"]["gender"] == "Men" else 1
        )
        st.session_state["loan_details"]["marital_status"] = st.selectbox(
            "Marital Status:", 
            ["Single", "Married"], 
            index=0 if st.session_state["loan_details"]["marital_status"] == "Single" else 1
        )
        st.session_state["loan_details"]["employee_status"] = st.selectbox(
            "Employment Status:", 
            ["employed", "self employed", "unemployed", "student"], 
            index=["employed", "self employed", "unemployed", "student"].index(
                st.session_state["loan_details"]["employee_status"]
            )
        )
        st.session_state["loan_details"]["residence_type"] = st.selectbox(
            "Residence Type:", 
            ["MORTGAGE", "OWN", "RENT"], 
            index=["MORTGAGE", "OWN", "RENT"].index(
                st.session_state["loan_details"]["residence_type"]
            )
        )
        st.session_state["loan_details"]["loan_purpose"] = st.selectbox(
            "Loan Purpose:", 
            ["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"], 
            index=["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"].index(
                st.session_state["loan_details"]["loan_purpose"]
            )
        )
    
    # EMI Calculator
    st.markdown("### Loan EMI Calculator")
    loan_amount = st.session_state["loan_details"]["loan_amount"]
    loan_term_years = st.session_state["loan_details"]["loan_term"] / 12
    interest_rate = st.number_input(
        "Interest Rate (%):", 
        min_value=0.1, max_value=15.0, step=0.1, 
        value=7.5
    )
    
    if st.button("Calculate EMI"):
        try:
            monthly_rate = interest_rate / (12 * 100)
            tenure_months = loan_term_years * 12
            if loan_amount > 0 and tenure_months > 0:
                emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / (
                    (1 + monthly_rate) ** tenure_months - 1
                )
                st.session_state["loan_details"]["emi"] = emi
                st.success(f"**Estimated EMI:** Rs. {emi:,.2f}")
            else:
                st.error("Please provide valid loan amount and term.")
        except Exception as e:
            st.error(f"Error calculating EMI: {str(e)}")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Previous", on_click=prev_step)
    with col2:
        st.button("Next", on_click=next_step)

# Step 3: Upload Documents
elif st.session_state["loan_details"]["current_step"] == 3:
    st.markdown("### Step 3: Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state["loan_details"]["id_proof"] = st.file_uploader(
            "Upload ID Proof (PDF, JPG, PNG)", 
            type=["pdf", "jpg", "jpeg", "png"],
            key="id_proof_uploader"
        )
    
    with col2:
        st.session_state["loan_details"]["address_proof"] = st.file_uploader(
            "Upload Address Proof (PDF, JPG, PNG)", 
            type=["pdf", "jpg", "jpeg", "png"],
            key="address_proof_uploader"
        )
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Previous", on_click=prev_step)
    with col2:
        st.button("Next", on_click=next_step)

# Step 4: Final Decision
elif st.session_state["loan_details"]["current_step"] == 4:
    st.markdown("### Step 4: Final Decision")
    loan_details = st.session_state["loan_details"]
    
    # Display summary
    st.markdown("#### Application Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Personal Information**")
        st.write(f"Name: {loan_details['full_name']}")
        st.write(f"Email: {loan_details['email']}")
        st.write(f"Phone: {loan_details['phone']}")
    
    with col2:
        st.markdown("**Loan Details**")
        st.write(f"Loan Amount: Rs. {loan_details['loan_amount']:,}")
        st.write(f"Loan Term: {loan_details['loan_term']} months")
        st.write(f"CIBIL Score: {loan_details['cibil_score']}")
    
    # Prepare input data for prediction
    try:
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
        
        # Ensure all required features are present
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
        
        # Prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        if prediction[0] == 1:
            st.markdown("### Loan Rejected ❌")
            st.error(f"Rejection Probability: {prediction_proba[0][1]:.2%}")
        else:
            st.markdown("### Loan Approved ✅")
            st.success(f"Approval Probability: {prediction_proba[0][0]:.2%}")
        
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
                pdf.cell(200, 10, txt=f"Full Name: {loan_details.get('full_name', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Email: {loan_details.get('email', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Phone: {loan_details.get('phone', 'N/A')}", ln=True)
                pdf.ln(5)
                
                # Loan Details
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(200, 10, txt="Loan Details:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"CIBIL Score: {loan_details.get('cibil_score', 'N/A')}", ln=True)
                pdf.cell(200, 10, txt=f"Annual Income: Rs. {loan_details.get('income_annum', 'N/A'):,}", ln=True)
                pdf.cell(200, 10, txt=f"Loan Amount: Rs. {loan_details.get('loan_amount', 'N/A'):,}", ln=True)
                pdf.cell(200, 10, txt=f"Loan Term: {loan_details.get('loan_term', 'N/A')} months", ln=True)
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
                label="📄 Download Report as PDF",
                data=pdf_buffer,
                file_name="loan_prediction_report.pdf",
                mime="application/pdf"
            )
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button("Previous", on_click=prev_step)
    with col2:
        if st.button("Submit Application"):
            st.success("Application submitted successfully!")
            # Reset form if needed
            # st.session_state["loan_details"] = {
            #     "full_name": "",
            #     "email": "",
            #     "phone": "",
            #     ... (rest of initial state)
            # }
            # st.session_state["loan_details"]["current_step"] = 1

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
    st.session_state["last_topic"] = None  # Track conversation topic
if "emi_active" not in st.session_state:
    st.session_state["emi_active"] = False  # Track EMI calculator trigger
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""  # Track the input field value

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
