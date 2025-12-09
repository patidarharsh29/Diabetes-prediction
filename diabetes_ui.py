import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        color: #ff4b4b;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff7f50);
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #ff7f50, #ff4b4b);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- LOTTIE LOADER --------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_health = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_tutvdkg0.json")

# -------------------- LOAD MODEL + ENCODERS + SCALER --------------------
model = joblib.load("model.joblib")
encoders = joblib.load("encoders.joblib")
scaler = joblib.load("scaler.joblib")

# -------------------- UI TITLE --------------------
st.title("🩺 Diabetes Prediction App ❤️")
st_lottie(lottie_health, height=200)

st.markdown("### 👉 Please enter your details")

# -------------------- INPUT FORM --------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])

with col2:
    smoking_history = st.selectbox("Smoking History",
                                   ["never", "current", "former", "ever", "not current", "No Info"])
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    hba1c = st.number_input("HbA1c Level", 3.0, 20.0, 5.5)
    blood_glucose = st.number_input("Blood Glucose Level", 50, 300, 100)

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict"):

    # ---- Create DataFrame (Correct ML Logic) ----
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c],
        "blood_glucose_level": [blood_glucose]
    })

    # ---- Encode categorical data ----
    for col in ["gender", "smoking_history"]:
        input_data[col] = encoders[col].transform(input_data[col])

    # ---- Scale numerical data ----
    numeric_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # ---- Column Order ----
    final_cols = ["gender", "age", "hypertension", "heart_disease",
                  "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

    input_data = input_data[final_cols]

    # ---- Prediction ----
    prediction = model.predict(input_data)[0]
    # prob = model.predict_proba(input_data)[0][1] * 100

    # ---- Result UI ----
    if prediction == 1:
        st.error(f"🚨 High Risk of Diabetes Detected!")
        st_lottie(load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json"), height=200)
    else:
        st.success(f"🎉 Low Risk of Diabetes!")
        st_lottie(load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_touohxv0.json"), height=200)


