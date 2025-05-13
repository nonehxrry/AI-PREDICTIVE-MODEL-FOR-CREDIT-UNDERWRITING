






























#I Love My India
#Harjit

import subprocess
import sys
import os

def install_package(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
        return True
    except ImportError:
        print(f"{package} not found. Installing...")
        try:
            install_dir = "."  # Install in the current directory (app root)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--target", install_dir, package])
            print(f"{package} installed successfully in {install_dir}")
            sys.path.insert(0, install_dir) # Add to Python path
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
        except FileNotFoundError:
            print(f"Error: pip not found.")
            return False

if not install_package("reportlab"):
    print("Failed to install reportlab. The app might not function correctly.")

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Image
from PIL import Image as PILImage
from transformers import pipeline
from langdetect import detect
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
            background: linear-gradient(to right, #f0f8ff, #e0ffff); /* Light and airy background */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }
        .header-container {
            background: linear-gradient(to right, #007bff, #6cace4); /* Vibrant blue header */
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .header-container h1 {
            font-size: 48px;
            margin-bottom: 10px;
        }
        .header-container p {
            font-size: 24px;
            font-style: italic;
        }
        .step-container {
            border: 1px solid #ddd;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 25px;
            background-color: #fff;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);
        }
        .step-title {
            font-size: 28px;
            color: #007bff;
            margin-bottom: 20px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .completed-tick {
            color: green;
            font-size: 1.5em;
            vertical-align: middle;
            margin-left: 10px;
        }
        .nav-button-container {
            margin-top: 30px;
            display: flex;
            gap: 20px;
            justify-content: flex-end;
        }
        .stButton>button {
            background-color: #28a745; /* Green button */
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #218838;
        }
        .stDownloadButton>button {
            background-color: #007bff; /* Blue download button */
        }
        .stDownloadButton>button:hover {
            background-color: #0056b3;
        }
        footer {
            text-align: center;
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 16px;
            color: #777;
        }
        .sidebar .stMarkdown h2 {
            color: #007bff;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .sidebar .stMarkdown h3 {
            color: #6cace4;
        }
        .sidebar .stNumberInput label,
        .sidebar .stSlider label,
        .sidebar .stSelectbox label {
            color: #333;
            font-weight: bold;
        }
        .sidebar .stButton>button {
            background-color: #28a745;
        }
        .sidebar .stButton>button:hover {
            background-color: #218838;
        }
        .stSuccess {
            color: #155724;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .stError {
            color: #721c24;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Header
st.markdown(
    """
    <div class="header-container">
        <h1>AI-Powered Credit Underwriting</h1>
        <p>Smarter Decisions, Faster Approvals</p>
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

# Initialize session state for steps
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0
if "step_complete" not in st.session_state:
    st.session_state["step_complete"] = [False] * 4

# Initialize session state for loan details
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

steps = ["Personal Information", "Loan Details", "Upload Documents", "Final Decision"]

def update_progress_bar():
    progress = (sum(st.session_state["step_complete"]) / len(steps))
    st.progress(progress)

update_progress_bar()

def next_step():
    if st.session_state["current_step"] < len(steps) - 1:
        st.session_state["current_step"] += 1

def prev_step():
    if st.session_state["current_step"] > 0:
        st.session_state["current_step"] -= 1

def mark_complete(step_index):
    st.session_state["step_complete"][step_index] = True
    update_progress_bar()

def clear_completion(step_index):
    st.session_state["step_complete"][step_index] = False
    update_progress_bar()

current_step_index = st.session_state["current_step"]
current_step_name = steps[current_step_index]
st.markdown(f"<h3 class='step-title'>{current_step_name} {'<span class=\"completed-tick\">✅</span>' if st.session_state['step_complete'][current_step_index] else ''}</h3>", unsafe_allow_html=True)

if current_step_name == "Personal Information":
    with st.container():
        st.markdown("#### Please provide your personal details:")
        st.session_state["loan_details"]["full_name"] = st.text_input("Full Name", st.session_state["loan_details"]["full_name"], key="full_name")
        st.session_state["loan_details"]["email"] = st.text_input("Email Address", st.session_state["loan_details"]["email"], key="email")
        st.session_state["loan_details"]["phone"] = st.text_input("Phone Number", st.session_state["loan_details"]["phone"], key="phone")
        if st.button("Mark as Complete", key="complete_personal"):
            mark_complete(0)
        st.markdown("<div class='nav-button-container'>", unsafe_allow_html=True)
        col_prev, col_next = st.columns(2)
        col_prev.button("Previous", on_click=prev_step, disabled=st.session_state["current_step"] == 0)
        col_next.button("Next", on_click=next_step, disabled=st.session_state["current_step"] == len(steps) - 1)
        st.markdown("</div>", unsafe_allow_html=True)
elif current_step_name == "Loan Details":
    with st.container(): # ONLY this, no arguments inside
        st.markdown("#### Enter the details of the loan you are seeking:")
        st.session_state["loan_details"]["cibil_score"] = st.slider("CIBIL Score (300-900):", 300, 900, st.session_state["loan_details"]["cibil_score"], key="cibil")
        st.session_state["loan_details"]["income_annum"] = st.number_input("Annual Income (INR):", min_value=0, step=10000, value=st.session_state["loan_details"]["income_annum"], key="income")
        st.session_state["loan_details"]["loan_amount"] = st.number_input("Loan Amount (INR):", min_value=0, step=10000, value=st.session_state["loan_details"]["loan_amount"], key="loan_amount")
        st.session_state["loan_details"]["loan_term"] = st.number_input("Loan Term (Months):", min_value=1, step=1, value=st.session_state["loan_details"]["loan_term"], key="loan_term")
        st.text_input("Loan Percent of Income (%):", value=f'{st.session_state["loan_details"]["loan_percent_income"]:.2f}', disabled=True)
        st.session_state["loan_details"]["active_loans"] = st.number_input("Number of Active Loans:", min_value=0, step=1, value=st.session_state["loan_details"]["active_loans"], key="active_loans")
        st.session_state["loan_details"]["gender"] = st.selectbox("Gender:", ["Men", "Women"], index=0 if st.session_state["loan_details"]["gender"] == "Men" else 1, key="gender")
        st.session_state["loan_details"]["marital_status"] = st.selectbox("Marital Status:", ["Single", "Married"], index=0 if st.session_state["loan_details"]["marital_status"] == "Single" else 1, key="marital_status")
        st.session_state["loan_details"]["employee_status"] = st.selectbox("Employment Status:", ["employed", "self employed", "unemployed", "student"], index=["employed", "self employed", "unemployed", "student"].index(st.session_state["loan_details"]["employee_status"]), key="employee_status")
        st.session_state["loan_details"]["residence_type"] = st.selectbox("Residence Type:", ["MORTGAGE", "OWN", "RENT"], index=["MORTGAGE", "OWN", "RENT"].index(st.session_state["loan_details"]["residence_type"]), key="residence_type")
        st.session_state["loan_details"]["loan_purpose"] = st.selectbox("Loan Purpose:", ["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"], index=["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"].index(st.session_state["loan_details"]["loan_purpose"]), key="loan_purpose")

        # EMI Calculator
        st.markdown("#### Estimated Monthly Installment (EMI)")
        loan_amount = st.session_state["loan_details"]["loan_amount"]
        loan_term_years = st.session_state["loan_details"]["loan_term"] / 12
        interest_rate = st.number_input("Annual Interest Rate (%):", min_value=0.1, max_value=25.0, step=0.1, value=7.5, key="interest_rate")
        monthly_rate = interest_rate / (12 * 100)
        tenure_months = loan_term_years * 12
        if loan_amount > 0 and tenure_months > 0 and monthly_rate > 0:
            emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
            st.session_state["loan_details"]["emi"] = emi
            st.write(f"**Estimated EMI:** Rs. {emi:,.2f}")
        else:
            st.session_state["loan_details"]["emi"] = None
            st.write("Please provide valid loan amount, term, and interest rate.")

        if st.button("Mark as Complete", key="complete_loan_details"):
            mark_complete(1)
        st.markdown("<div class='nav-button-container'>", unsafe_allow_html=True)
        col_prev, col_next = st.columns(2)
        col_prev.button("Previous", on_click=prev_step, disabled=st.session_state["current_step"] == 0)
        col_next.button("Next", on_click=next_step, disabled=st.session_state["current_step"] == len(steps) - 1)
        st.markdown("</div>", unsafe_allow_html=True)
elif current_step_name == "Upload Documents":
    with st.container(): # Remove border=True and class_="step-container"
        st.markdown("#### Please upload the required documents:")
        st.session_state["loan_details"]["id_proof"] = st.file_uploader("Upload ID Proof (e.g., Aadhaar, Passport)", type=["png", "jpg", "jpeg"], key="id_proof")
        st.session_state["loan_details"]["address_proof"] = st.file_uploader("Upload Address Proof (e.g., Utility Bill, Bank Statement)", type=["png", "jpg", "jpeg"], key="address_proof")
        if st.button("Mark as Complete", key="complete_upload"):
            mark_complete(2)
        st.markdown("<div class='nav-button-container'>", unsafe_allow_html=True)
        col_prev, col_next = st.columns(2)
        col_prev.button("Previous", on_click=prev_step, disabled=st.session_state["current_step"] == 0)
        col_next.button("Next", on_click=next_step, disabled=st.session_state["current_step"] == len(steps) - 1)
        st.markdown("</div>", unsafe_allow_html=True)
elif current_step_name == "Final Decision":
    with st.container():
        st.markdown("#### Review your details and get the decision:")
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
                st.markdown("<h3 style='color:red;'>Loan Rejected ❌</h3>", unsafe_allow_html=True)
                st.error(f"Rejection Probability: {prediction_proba[0][1]:.2f}")
            else:
                st.markdown("<h3 style='color:green;'>Loan Approved ✅</h3>", unsafe_allow_html=True)
                st.success(f"Approval Probability: {prediction_proba[0][0]:.2f}")

            # Generate PDF Report using reportlab
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            title_style = styles["h1"]
            heading_style = styles["h2"]
            normal_style = styles.get("Normal", ParagraphStyle(name='Normal', fontName='Helvetica', fontSize=10))

            def draw_underlined_heading(canvas, text, style, x, y):
                p = Paragraph(text, style)
                w, h = p.wrapOn(canvas, letter[0] - 2 * inch, 50)
                p.drawOn(canvas, x, y)
                canvas.line(x, y - 2, x + w, y - 2)
                return h

            # Title
            title = Paragraph("Loan Application Decision Report", title_style)
            title.wrapOn(c, letter[0] - 2 * inch, 50)
            title.drawOn(c, inch, letter[1] - inch - 20)
            current_y = letter[1] - inch - 50

            # Border
            c.rect(inch, inch, letter[0] - 2 * inch, letter[1] - 2 * inch)

            # Personal Information
            heading_height = draw_underlined_heading(c, "Personal Information", heading_style, inch, current_y)
            current_y -= heading_height + 0.2 * inch
            c.line(inch, current_y, letter[0] - inch, current_y)
            current_y -= 0.1 * inch

            p = Paragraph(f"<b>Full Name:</b> {loan_details.get('full_name', 'N/A')}", normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.2 * inch

            p = Paragraph(f"<b>Email:</b> {loan_details.get('email', 'N/A')}", normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.2 * inch

            p = Paragraph(f"<b>Phone:</b> {loan_details.get('phone', 'N/A')}", normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.3 * inch

            # Loan Details
            heading_height = draw_underlined_heading(c, "Loan Details", heading_style, inch, current_y)
            current_y -= heading_height + 0.2 * inch
            c.line(inch, current_y, letter[0] - inch, current_y)
            current_y -= 0.1 * inch

            p = Paragraph(f"<b>CIBIL Score:</b> {loan_details.get('cibil_score', 'N/A')}", normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.2 * inch

            p = Paragraph(f"<b>Loan Amount:</b> Rs. {loan_details.get('loan_amount', 'N/A'):,.2f}", normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.2 * inch

            p = Paragraph(f"<b>Loan Term:</b> {loan_details.get('loan_term', 'N/A')} months", normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.2 * inch

            emi_value = loan_details.get("emi", None)
            emi_text = f"<b>Estimated EMI:</b> Rs. {emi_value:,.2f}" if emi_value is not None else "<b>Estimated EMI:</b> Not Calculated"
            p = Paragraph(emi_text, normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.3 * inch

            # Prediction Results
            heading_height = draw_underlined_heading(c, "Prediction Results", heading_style, inch, current_y)
            current_y -= heading_height + 0.2 * inch
            c.line(inch, current_y, letter[0] - inch, current_y)
            current_y -= 0.1 * inch

            decision_text = f"<b>Decision:</b> {'Approved' if prediction[0] == 0 else 'Rejected'}"
            p = Paragraph(decision_text, normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.2 * inch

            approval_prob_text = f"<b>Approval Probability:</b> {prediction_proba[0][0]:.2f}"
            p = Paragraph(approval_prob_text, normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.2 * inch

            rejection_prob_text = f"<b>Rejection Probability:</b> {prediction_proba[0][1]:.2f}"
            p = Paragraph(rejection_prob_text, normal_style)
            p.wrapOn(c, letter[0] - 2 * inch, 50)
            p.drawOn(c, inch, current_y)
            current_y -= 0.3 * inch

            # Add Images (Adjusted Positioning)
            image_y_position = letter[1] - 2 * inch - 20 # Start below the border
            image_x_start = inch + 20
            image_width = 100
            image_height = 75
            image_offset_x = 120
            image_text_offset_y = 10

            c.setFont("Helvetica", 8) # Set font for image labels

            if loan_details["id_proof"] is not None:
                try:
                    img_bytes = loan_details["id_proof"].getvalue()
                    pil_img = PILImage.open(BytesIO(img_bytes))
                    img_buffer = BytesIO()
                    pil_img.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    c.drawImage(img_buffer, image_x_start, image_y_position - image_height, width=image_width, height=image_height, preserveAspectRatio=True)
                    c.drawString(image_x_start, image_y_position - image_height - image_text_offset_y, "ID Proof")
                    image_x_start += image_offset_x
                except Exception as e:
                    st.error(f"Error adding ID Proof to PDF: {e}")

            if loan_details["address_proof"] is not None and image_x_start < letter[0] - inch - image_width:
                try:
                    img_bytes = loan_details["address_proof"].getvalue()
                    pil_img = PILImage.open(BytesIO(img_bytes))
                    img_buffer = BytesIO()
                    pil_img.save(img_buffer, format="PNG")
                    img_buffer.seek(0)
                    c.drawImage(img_buffer, image_x_start, image_y_position - image_height, width=image_width, height=image_height, preserveAspectRatio=True)
                    c.drawString(image_x_start, image_y_position - image_height - image_text_offset_y, "Address Proof")
                except Exception as e:
                    st.error(f"Error adding Address Proof to PDF: {e}")

            c.save()
            buffer.seek(0)

            st.download_button(
                label="Download Report as PDF",
                data=buffer,
                file_name="loan_prediction_report.pdf",
                mime="application/pdf",
                key="pdf_download_button"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

        st.markdown("<div class='nav-button-container'>", unsafe_allow_html=True)
        col_prev, col_next = st.columns(2)
        col_prev.button("Previous", on_click=prev_step, disabled=st.session_state["current_step"] == 0)
        col_next.button("Next", on_click=next_step, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if st.button("Submit Application", key="submit_application_final"):
            st.success("Loan application submitted successfully!")
            mark_complete(3)
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
st.sidebar.markdown("## 🤖 AI Financial Chatbot & EMI Calculator")

# --- Initialize Chat History & Session Variables ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "bot", "content": "👋 Hello! How can I help you with your loan application or financial queries today?\n\n**Explore these topics:**\n- 🏦 Loan Assistance\n- 💰 EMI Calculation\n- 📊 Investment Insights\n- ❓ General Finance Questions"}
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

    # Standard Greetings
    greetings = ["hello", "hi", "hey", "how are you"]
    if user_message in greetings:
        return "👋 Hello! How can I assist you today?"

    # Loan Assistance
    loan_keywords = ["loan", "borrow", "credit", "finance"]
    if any(keyword in user_message for keyword in loan_keywords):
        st.session_state["last_topic"] = "loan_assistance"
        return "🏦 I can help with information on different types of loans, eligibility criteria, and the application process. What specific type of loan are you interested in (e.g., personal loan, home loan, business loan)?"

    if st.session_state["last_topic"] == "loan_assistance":
        if "personal" in user_message:
            return "📌 **Personal Loans:** Typically unsecured loans for various personal needs. Interest rates vary based on credit score and lender. Loan amounts can range from a few thousand to several lakhs with tenures from 1 to 5 years."
        elif "home" in user_message:
            return "🏡 **Home Loans:** Secured loans to purchase or construct a property. Interest rates are usually lower than personal loans, with tenures extending up to 30 years. Eligibility depends on income, credit score, and property value."
        elif "business" in user_message:
            return "💼 **Business Loans:** Designed to finance business needs like expansion, working capital, or equipment purchase. Various types are available, including term loans, working capital loans, and equipment financing."

    # EMI Calculator Activation
    emi_keywords = ["emi", "calculate monthly payment", "monthly installment"]
    if any(keyword in user_message for keyword in emi_keywords):
        st.session_state["emi_active"] = True
        st.session_state["last_topic"] = "emi_calculator"
        return "💰 Sure, let's calculate your EMI. Please provide the loan amount, annual interest rate (in percentage), and loan tenure (in years)."

    if st.session_state["emi_active"] and st.session_state["last_topic"] == "emi_calculator":
        try:
            parts = user_message.split()
            loan_amount = None
            interest_rate = None
            tenure = None
            for i, part in enumerate(parts):
                if part.lower() == "amount" and i + 1 < len(parts):
                    loan_amount = float(parts[i + 1])
                elif part.lower() == "rate" and i + 1 < len(parts):
                    interest_rate = float(parts[i + 1])
                elif part.lower() == "year" and i + 1 < len(parts):
                    tenure = float(parts[i + 1])

            if loan_amount is not None and interest_rate is not None and tenure is not None:
                monthly_rate = interest_rate / (12 * 100)
                tenure_months = tenure * 12
                emi = round((loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1), 2)
                st.session_state["emi_active"] = False
                st.session_state["last_topic"] = None
                return f"📊 Your estimated monthly EMI is: Rs. {emi:,.2f}"
            else:
                return "Please provide the loan amount, annual interest rate, and loan tenure in years."
        except ValueError:
            return "Invalid input. Please provide numeric values for loan amount, interest rate, and tenure."

    # Investment Insights
    investment_keywords = ["invest", "investment", "stocks", "bonds", "mutual funds"]
    if any(keyword in user_message for keyword in investment_keywords):
        st.session_state["last_topic"] = "investment_insights"
        return "📊 I can provide some general information on investment options like stocks, bonds, and mutual funds. Please remember that I am not a financial advisor, and this is not financial advice."

    if st.session_state["last_topic"] == "investment_insights":
        if "stocks" in user_message:
            return "📈 **Stocks:** Represent ownership in a company. They can offer high growth potential but also come with higher risk."
        elif "bonds" in user_message:
            return "📉 **Bonds:** Represent debt issued by governments or corporations. They are generally considered less risky than stocks but offer lower returns."
        elif "mutual funds" in user_message:
            return "💰 **Mutual Funds:** Pools of money invested in a diversified portfolio of stocks, bonds, or other assets, managed by a professional fund manager."

    # General Finance Questions
    finance_keywords = ["finance", "financial", "money", "budget", "saving"]
    if any(keyword in user_message for keyword in finance_keywords):
        st.session_state["last_topic"] = "general_finance"
        return "❓ I can try to answer general finance questions. What's on your mind?"

    # Default Response
    return "🤔 I'm sorry, I didn't quite understand that. Can you please rephrase your question or choose from the topics mentioned above?"

# --- Display Chat History ---
st.sidebar.markdown("## 💬 Chat with our AI Assistant")
for message in st.session_state["chat_messages"]:
    role = "👤 You" if message["role"] == "user" else "🤖 AI Assistant"
    st.sidebar.markdown(f"**{role}:** {message['content']}")

# --- Text Input Field for Manual Chat ---
user_input = st.sidebar.text_input("Ask a question:", value=st.session_state["user_input"], key="chat_input")

# --- Process User Input ---
if st.sidebar.button("Send", key="send_chat"):
    if user_input.strip():
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        bot_reply = chatbot_response(user_input)
        st.session_state["chat_messages"].append({"role": "bot", "content": bot_reply})
        st.session_state["user_input"] = ""
        st.rerun()

# --- EMI Calculator in Sidebar ---
if st.session_state["emi_active"]:
    st.sidebar.markdown("### 💰 EMI Calculator")
    loan_amount_chat = st.sidebar.number_input("Loan Amount (₹)", min_value=1000, value=500000, step=1000, key="emi_amount_chat")
    interest_rate_chat = st.sidebar.number_input("Annual Interest Rate (%)", min_value=0.1, value=10.0, step=0.1, key="emi_rate_chat")
    tenure_chat = st.sidebar.number_input("Loan Tenure (Years)", min_value=1, value=5, step=1, key="emi_tenure_chat")

    if st.sidebar.button("Calculate EMI", key="calculate_emi_chat"):
        monthly_rate_chat = interest_rate_chat / (12 * 100)
        tenure_months_chat = tenure_chat * 12
        if loan_amount_chat > 0 and tenure_months_chat > 0 and monthly_rate_chat > 0:
            emi_result_chat = round((loan_amount_chat * monthly_rate_chat * (1 + monthly_rate_chat) ** tenure_months_chat) / ((1 + monthly_rate_chat) ** tenure_months_chat - 1), 2)
            st.sidebar.success(f"📌 Your Monthly EMI: ₹{emi_result_chat:,.2f}")
        else:
            st.sidebar.error("Please enter valid loan details for EMI calculation.")
        st.session_state["emi_active"] = False
        st.session_state["last_topic"] = None
