import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('weighted method/models/rf_model_weighted.pkl')

# Title for your web app
st.title("Loan Default Prediction")

# Create input fields for user input
Client_Income = st.number_input("Client Income", min_value=0)
Car_Owned = st.selectbox("Car Owned", [0, 1])  # 0 = No, 1 = Yes
Bike_Owned = st.selectbox("Bike Owned", [0, 1])  # 0 = No, 1 = Yes
Active_Loan = st.selectbox("Active Loan", [0, 1])  # 0 = No, 1 = Yes
House_Own = st.selectbox("House Owned", [0, 1])  # 0 = No, 1 = Yes
Child_Count = st.number_input("Child Count", min_value=0)
Credit_Amount = st.number_input("Credit Amount", min_value=0)
Loan_Annuity = st.number_input("Loan Annuity", min_value=0)

# User inputs for categorical variables
Client_Education = st.selectbox("Client Education", ['Graduation', 'Gradudation dropout', 'Junior secondary', 'Post Grad', 'Secondary'])
Client_Income_Type = st.selectbox("Client Income Type", ['Commercial', 'Govt Job', 'Maternity leave', 'Retired', 'Service', 'Student'])
Client_Marital_Status = st.selectbox("Client Marital Status", ['D', 'M', 'S', 'W'])
Client_Gender = st.selectbox("Client Gender", ['Female', 'Male'])
Loan_Contract_Type = st.selectbox("Loan Contract Type", ['CL', 'RL'])
Workphone_Working = st.selectbox("Workphone Working", [0, 1])  # 0 = No, 1 = Yes
Client_Family_Members = st.number_input("Client Family Members", min_value=0)
Age_Years = st.number_input("Age (Years)", min_value=0)
Employed_Years = st.number_input("Employed (Years)", min_value=0)

# Define a function to map Client_Education to its label-encoded value
def encode_client_education(education):
    education_mapping = {
        'Graduation': 0,
        'Graduation dropout': 1,
        'Junior secondary': 2,
        'Post Grad': 3,
        'Secondary': 4
    }
    return education_mapping.get(education, -1)  # Return -1 if not found

# Encode categorical variables into one-hot encoding
def encode_input():
    # Map Client_Education to its label-encoded value
    education_encoded = encode_client_education(Client_Education)

    # One-hot encoding for Client_Income_Type
    income_type_encoded = [
        1 if Client_Income_Type == "Commercial" else 0,
        1 if Client_Income_Type == "Govt Job" else 0,
        1 if Client_Income_Type == "Maternity leave" else 0,
        1 if Client_Income_Type == "Retired" else 0,
        1 if Client_Income_Type == "Service" else 0,
        1 if Client_Income_Type == "Student" else 0,
    ]

    # One-hot encoding for Client_Marital_Status
    marital_status_encoded = [
        1 if Client_Marital_Status == "D" else 0,
        1 if Client_Marital_Status == "M" else 0,
        1 if Client_Marital_Status == "S" else 0,
        1 if Client_Marital_Status == "W" else 0,
    ]

    # One-hot encoding for Client_Gender
    gender_encoded = [
        1 if Client_Gender == "Female" else 0,
        1 if Client_Gender == "Male" else 0,
    ]

    # One-hot encoding for Loan_Contract_Type
    loan_contract_encoded = [
        1 if Loan_Contract_Type == "CL" else 0,
        1 if Loan_Contract_Type == "RL" else 0,
    ]

    # Combine all features into one array
    input_features = [
        Client_Income, Car_Owned, Bike_Owned, Active_Loan, House_Own, Child_Count,
        Credit_Amount, Loan_Annuity, Workphone_Working, Client_Family_Members, Age_Years, Employed_Years,
        education_encoded  # Use the label-encoded value directly
    ] + income_type_encoded + marital_status_encoded + gender_encoded + loan_contract_encoded


    return np.array(input_features).reshape(1, -1)


# Button to predict
if st.button("Predict"):
    # Preprocess input
    input_data = encode_input()

    # Predict using the model
    prediction = model.predict(input_data)

    # Display result
    if prediction[0] == 0:
        st.success("The client is predicted to NOT default on the loan.")
    else:
        st.error("The client is predicted to DEFAULT on the loan.")






