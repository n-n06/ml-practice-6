import requests
import streamlit as st

API_URL = "http://api:8000/predict"

st.set_page_config(
    page_title="Deposit Predictor",
    page_icon="🏦",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background-color: black;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Page title area */
.hero {
    background: #1A1A2E;
    border-radius: 16px;
    padding: 2rem 2.5rem 1.75rem;
    margin-bottom: 2rem;
    color: #F7F4EF;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin: 0 0 0.3rem 0;
    color: #F7F4EF;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 0.95rem;
    color: #A0A0B8;
    margin: 0;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin: 1.75rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #E0DBD3;
}

/* Result cards */
.result-card {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.25rem;
}
.result-yes {
    background: #E8F5E9;
    border: 1.5px solid #A5D6A7;
}
.result-no {
    background: #FBE9E7;
    border: 1.5px solid #FFAB91;
}
.result-icon { font-size: 2.4rem; line-height: 1; }
.result-title {
    color: black;
    font-size: 1.4rem;
    margin: 0 0 0.2rem 0;
}
.result-sub { font-size: 0.88rem; color: #555; margin: 0; font-weight: 300; }

/* Probability meter label */
.prob-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #555;
    margin-bottom: 0.25rem;
}

/* Submit button */
div.stButton > button {
    background: #1A1A2E;
    color: #F7F4EF;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    cursor: pointer;
    transition: background 0.2s;
    width: 100%;
    margin-top: 1rem;
}
div.stButton > button:hover {
    background: #2D2D50;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🏦 Deposit Predictor</h1>
    <p>Enter client data below to predict term deposit subscription likelihood.</p>
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):

    # Client profile
    st.markdown('<div class="section-label">Client Profile</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=40)
    with col2:
        job = st.selectbox("Job", [
            "management", "blue-collar", "technician", "admin.", "services",
            "retired", "self-employed", "student", "unemployed",
            "entrepreneur", "housemaid", "unknown",
        ])
    with col3:
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])

    col4, col5 = st.columns(2)
    with col4:
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    with col5:
        balance = st.number_input("Account Balance (€)", value=1000, step=100)

    # Financial flags
    st.markdown('<div class="section-label">Financial Flags</div>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    with col6:
        default = st.selectbox("Credit Default", ["no", "yes"])
    with col7:
        housing = st.selectbox("Housing Loan", ["no", "yes"])
    with col8:
        loan = st.selectbox("Personal Loan", ["no", "yes"])

    # Last contact
    st.markdown('<div class="section-label">Last Contact</div>', unsafe_allow_html=True)
    col9, col10, col11 = st.columns(3)
    with col9:
        contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    with col10:
        month = st.selectbox("Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ])
    with col11:
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)

    col12, col13 = st.columns(2)
    with col12:
        duration = st.number_input("Call Duration (sec)", min_value=0, value=180)
    with col13:
        campaign = st.number_input("Contacts This Campaign", min_value=0, value=1)

    # Previous campaign
    st.markdown('<div class="section-label">Previous Campaign</div>', unsafe_allow_html=True)
    col14, col15, col16 = st.columns(3)
    with col14:
        pdays = st.number_input("Days Since Last Contact", min_value=0, value=0)
    with col15:
        previous = st.number_input("Previous Contacts", min_value=0, value=0)
    with col16:
        poutcome = st.selectbox("Previous Outcome", ["unknown", "success", "failure", "other"])

    submitted = st.form_submit_button("Predict")

# Prediction 
if submitted:
    payload = {
        "age": age, "job": job, "marital": marital,
        "education": education, "default": default, "balance": balance,
        "housing": housing, "loan": loan, "contact": contact,
        "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays,
        "previous": previous, "poutcome": poutcome,
    }

    with st.spinner("Running prediction..."):
        try:
            resp = requests.post(API_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            prediction = data["deposit_prediction"]
            probability = data["deposit_probability"]

            if prediction == 1:
                st.markdown(f"""
                <div class="result-card result-yes">
                    <div class="result-icon">✅</div>
                    <div>
                        <p class="result-title">Likely to Subscribe</p>
                        <p class="result-sub">This client is predicted to open a term deposit.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-no">
                    <div class="result-icon">❌</div>
                    <div>
                        <p class="result-title">Unlikely to Subscribe</p>
                        <p class="result-sub">This client is predicted to decline a term deposit.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="prob-label">Subscription Probability</div>', unsafe_allow_html=True)
            st.progress(probability, text=f"{probability:.1%}")

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot reach the API. Make sure your FastAPI server is running on `api:8000`.")
        except requests.exceptions.HTTPError as e:
            st.error(f"⚠️ API returned an error: {e.response.status_code} — {e.response.text}")
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
