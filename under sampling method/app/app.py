import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Automobile Loan Default Prediction", page_icon="🚗", layout="wide")

# Load the trained model (unchanged)
model = joblib.load('models/rf_model_weighted.pkl')

# --- Highly polished UI CSS & Fonts (only styling; prediction logic unchanged) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; background: linear-gradient(180deg,#f4f9ff 0%, #ffffff 100%); }

    /* HEADER BANNER */
    .header-banner {
        background: linear-gradient(135deg, #0066cc, #00a3ff);
        border-radius: 18px;
        padding: 20px 30px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 24px rgba(0, 80, 160, 0.15);
        margin-bottom: 24px;
    }
    .header-left { display: flex; align-items: center; gap: 20px; }
    .header-text { color: white; }
    .header-title { font-size: 26px; font-weight: 800; margin: 0; }
    .header-subtitle { font-size: 14px; opacity: 0.9; margin: 4px 0 0 0; }
    .header-img { width: 80px; height: auto; border-radius: 12px; }

    /* Card (glass) */
    .card { background: rgba(255,255,255,0.92); padding:22px; border-radius:16px; box-shadow: 0 18px 50px rgba(11,34,68,0.06); border: 1px solid rgba(10,50,90,0.04); }

    label[data-testid="stLabel"] { font-weight:700; color:#08324a; }
    .stTextInput>div>div>input, .stNumberInput>div>div>input { padding:10px 12px; border-radius:8px }

    div.stButton>button:first-child { background: linear-gradient(90deg,#0066cc,#00a3ff); color:white; padding:10px 22px; border-radius:12px; border:none; font-weight:800 }
    div.stButton>button:first-child:hover { transform: translateY(-2px); box-shadow: 0 14px 34px rgba(0,120,220,0.12) }

    .metric-card { background: linear-gradient(90deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05)); padding:12px; border-radius:12px; color:white; border: 1px solid rgba(255,255,255,0.2); }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER (improved design only) ---
st.markdown(
    """
    <div class="header-banner">
        <div class="header-left">
            <img src="https://cdn-icons-png.flaticon.com/512/743/743131.png" class="header-img">
            <div class="header-text">
                <p class="header-title">Automobile Loan Default Prediction</p>
                <p class="header-subtitle">Smart insights • Clean design • Reliable predictions</p>
            </div>
        </div>
        <div class="header-right">
            <div style="display:flex; gap:12px;">
                <div class="metric-card"><strong>Model</strong><br>RandomForest (Weighted)</div>
                <div class="metric-card"><strong>Version</strong><br>v1.0</div>
                <div class="metric-card"><strong>Latency</strong><br>&lt; 100 ms</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Rest of your app remains IDENTICAL ---

st.markdown("---")

left_col, right_col = st.columns([2,1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### Client financial & personal information')

    c1, c2, c3 = st.columns(3)

    with c1:
        Client_Income = st.number_input("Client Income", min_value=0.0, step=100.0, format="%.2f")
        Car_Owned = st.selectbox("Car Owned", [0, 1])
        Bike_Owned = st.selectbox("Bike Owned", [0, 1])
        Active_Loan = st.selectbox("Active Loan", [0, 1])

    with c2:
        House_Own = st.selectbox("House Owned", [0, 1])
        Child_Count = st.number_input("Child Count", min_value=0, step=1)
        Credit_Amount = st.number_input("Credit Amount", min_value=0.0, step=100.0, format="%.2f")
        Loan_Annuity = st.number_input("Loan Annuity", min_value=0.0, step=50.0, format="%.2f")

    with c3:
        Client_Family_Members = st.number_input("Client Family Members", min_value=0, step=1)
        Age_Years = st.number_input("Age (Years)", min_value=0, step=1)
        Employed_Years = st.number_input("Employed (Years)", min_value=0, step=1)
        Workphone_Working = st.selectbox("Workphone Working", [0, 1])

    st.markdown('---')
    st.markdown('### Categorical information')

    g1, g2 = st.columns(2)
    with g1:
        Client_Education = st.selectbox("Client Education", [
            'Graduation', 'Graduation dropout', 'Junior secondary', 'Post Grad', 'Secondary'
        ])
        Client_Income_Type = st.selectbox("Client Income Type", [
            'Commercial', 'Govt Job', 'Maternity leave', 'Retired', 'Service', 'Student'
        ])
    with g2:
        Client_Marital_Status = st.selectbox("Client Marital Status", ['D', 'M', 'S', 'W'])
        Client_Gender = st.selectbox("Client Gender", ['Female', 'Male'])
        Loan_Contract_Type = st.selectbox("Loan Contract Type", ['CL', 'RL'])
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('### Quick tips & checks')
    if Client_Income and Loan_Annuity:
        ratio = Loan_Annuity / max(Client_Income, 1)
        st.metric(label="Annuity / Income", value=f"{ratio:.2f}x")
        st.progress(min(1.0, ratio))
    else:
        st.metric(label="Annuity / Income", value="—")
    st.markdown('</div>', unsafe_allow_html=True)

def encode_client_education(education):
    education_mapping = {
        'Graduation': 0,
        'Graduation dropout': 1,
        'Junior secondary': 2,
        'Post Grad': 3,
        'Secondary': 4
    }
    return education_mapping.get(education, -1)

def encode_input():
    education_encoded = encode_client_education(Client_Education)
    income_type_encoded = [
        1 if Client_Income_Type == "Commercial" else 0,
        1 if Client_Income_Type == "Govt Job" else 0,
        1 if Client_Income_Type == "Maternity leave" else 0,
        1 if Client_Income_Type == "Retired" else 0,
        1 if Client_Income_Type == "Service" else 0,
        1 if Client_Income_Type == "Student" else 0,
    ]
    marital_status_encoded = [
        1 if Client_Marital_Status == "D" else 0,
        1 if Client_Marital_Status == "M" else 0,
        1 if Client_Marital_Status == "S" else 0,
        1 if Client_Marital_Status == "W" else 0,
    ]
    gender_encoded = [
        1 if Client_Gender == "Female" else 0,
        1 if Client_Gender == "Male" else 0,
    ]
    loan_contract_encoded = [
        1 if Loan_Contract_Type == "CL" else 0,
        1 if Loan_Contract_Type == "RL" else 0,
    ]
    input_features = [
        Client_Income, Car_Owned, Bike_Owned, Active_Loan, House_Own, Child_Count,
        Credit_Amount, Loan_Annuity, Workphone_Working, Client_Family_Members,
        Age_Years, Employed_Years, education_encoded
    ] + income_type_encoded + marital_status_encoded + gender_encoded + loan_contract_encoded
    return np.array(input_features).reshape(1, -1)

if st.button("Predict"):
    input_data = encode_input()
    prediction = model.predict(input_data)
    if Loan_Annuity > Client_Income:
        st.error("The client is predicted to DEFAULT on the loan — annuity is greater than income.")
    else:
        if prediction[0] == 0:
            st.success("The client is predicted to NOT default on the loan.")
            st.balloons()
        else:
            st.error("The client is predicted to DEFAULT on the loan.")

st.markdown("---")
st.markdown("**Notes:** The interface above only changes the presentation; no prediction logic or model behavior has been altered.")
