# AI 

This project is a **Streamlit-based AI Predictive Methods for Credit Underwriting** application that utilizes a machine learning model to predict whether a loan application will be **approved or rejected** based on user-provided inputs. The app also generates a **downloadable PDF report** with detailed results and insights.

## Presentation Link 
https://www.canva.com/design/DAGd4QHNyMw/PqKP2SuqxH5LwEoZloEbwg/view?utm_content=DAGd4QHNyMw&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h0d00988ec0

## Features

- **Interactive Web App**: Built with Streamlit for an intuitive and responsive user interface.
- **User Inputs**: Fields for CIBIL score, income, loan amount, loan term, and more.
- **Prediction Model**: Uses a pre-trained machine learning model to determine loan approval status.
- **Downloadable Report**: Generates a professional **PDF report** summarizing predictions and user inputs.
- **EMI Calculator**: Calculate monthly EMI based on loan amount, interest rate, and term.
- **AI Chatbot**: Provides financial advice and helps users with loan-related queries.
- **Enhanced Styling**: Custom CSS for a better user experience.

## Technologies Used

- **Python**: Programming language for building the application.
- **Streamlit**: Framework for creating the interactive web app.
- **FPDF**: Library for generating downloadable PDF reports.
- **pandas**: For data preparation and handling user inputs.
- **matplotlib**: For optional visualizations.
- **joblib**: For loading the pre-trained machine learning model.
- **transformers**: For NLP-based chatbot responses (using pre-trained models).
- **langdetect**: For language detection (used in the chatbot).

## Prerequisites

Ensure you have the following installed on your system:

- **Python 3.7+**
- **pip (Python package manager)**

### Install Dependencies
Run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```

### `requirements.txt` File Contains:
```
streamlit
pandas
matplotlib
joblib
fpdf2
transformers
langdetect
```

## How to Run

### Clone the Repository:
```bash
git clone https://github.com/nonehxrry/AI-PREDICTIVE-MODEL-FOR-CREDIT-UNDERWRITING/
cd AI-Predictive-Methods-for-Credit-underwriting
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Run the Application:
```bash
streamlit run streamlit_app.py
```

Open the provided **local URL** in your browser to access the app.

## File Structure

```
├── streamlit_app.py  # Main application file
├── requirements.txt  # List of required Python libraries
├── best_features_model.pkl  # Pre-trained machine learning model file
├── FreeSerif.ttf  # Font for generating PDF reports (ensure this font is available)
```

## User Inputs

The app provides the following input fields in the sidebar:

- **CIBIL Score** (300-900)
- **Annual Income** (INR)
- **Loan Amount** (INR)
- **Loan Term** (months)
- **Number of Active Loans**
- **Gender**
- **Marital Status**
- **Employment Status**
- **Residence Type**
- **Loan Purpose**

## Output

- **Loan Status**: Displays whether the loan is **Approved** or **Rejected**.
- **Prediction Probabilities**: Shows the probability of approval and rejection.
- **Downloadable PDF Report**:
  - Prediction results.
  - Input details provided by the user.
  - Summary of probabilities.

## Deployment

The application is live and can be accessed at:
[https://ai-predictive-model-for-credit-underwriting-lbfypnypbmpy8ixieq.streamlit.app/)

To deploy the app on **Streamlit Cloud**:

To deploy the app on **Streamlit Cloud**:

1. **Push the project to GitHub.**
2. **Connect your GitHub repository to Streamlit Cloud.**
3. **Ensure `requirements.txt` is present for dependency installation.**
4. **Deploy and access your app via the Streamlit Cloud link.**

## Troubleshooting

### Missing Library Error
If you encounter an error like `ModuleNotFoundError: No module named 'fpdf'`, install it manually:
```bash
pip install fpdf
```

### Unicode Character Error (₹ Symbol)
Ensure you have a Unicode-compatible font (e.g., `FreeSerif.ttf`) in your working directory. Update the PDF font registration in the code if necessary.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Contributors

- **Harjit Singh Bhadauriya** (Developer)
- [GitHub Profile](https://github.com/nonehxrry)

## Feedback

For any issues or suggestions, please open an **issue** on the [GitHub repository](https://github.com/nonehxrry/AI-PREDICTIVE-MODEL-FOR-CREDIT-UNDERWRITING/issues).

