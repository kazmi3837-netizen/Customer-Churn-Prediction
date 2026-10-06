import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("churn_preprocessor.pkl")


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")
st.markdown(
    "### Predict whether a customer is likely to leave the service"
)

st.divider()


# --------------------------------------------------
# Customer Information
# --------------------------------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col3:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


# --------------------------------------------------
# Service Information
# --------------------------------------------------

st.subheader("🌐 Service Information")

col1, col2, col3 = st.columns(3)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col2:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )


# --------------------------------------------------
# Billing Information
# --------------------------------------------------

st.subheader("💳 Billing Information")

col1, col2, col3 = st.columns(3)

with col1:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col3:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=monthly_charges * tenure
)


st.divider()


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Transform input
    customer_encoded = preprocessor.transform(customer_data)

    # Prediction
    prediction = model.predict(customer_encoded)[0]

    probability = model.predict_proba(customer_encoded)[0][1]

    st.subheader("📌 Prediction Result")

    if prediction == 1:

        st.error("🔴 Customer is likely to CHURN")

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    else:

        st.success("🟢 Customer is likely to STAY")

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )