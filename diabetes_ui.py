# import streamlit as st
# import joblib
# import pandas as pd


# model  = joblib.load("Random.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_columns = joblib.load("columns.pkl")




# st.title("Diabetes prediction by harsh ❤️")
# st.markdown("Provide the following details")
# Age = st.slider("Age",18,100,40)
# Gender = st.selectbox("SEX",['M','F'])
# Hypertension = st.selectbox("Hyper Tension",[0,1])
# Heart_disease = st.selectbox("Heart Disease",[0,1])
# Smoking_history = st.selectbox("Smoking History",['Never','No Info','Current','Former','Ever','Not Current'])
# Bmi = st.number_input("Body Mass Index",10.0,100.0,20.0)
# HbA1c_level = st.number_input("Hemoglobin A1c test",4.0,16.0,5.0)
# Blood_glucose_level = st.number_input("Blood Glucose Level",50,300,120)




# if st.button('Predict'):
#     raw_input = {
#         'Age': Age,
#         'Gender'+ Gender: 1,
#         'Hypertension': Hypertension,
#         'Heart_disease' :Heart_disease,
#         'Smoking_history' : Smoking_history,
#         'Bmi' : Bmi,
#         'HbA1c_level' : HbA1c_level,
#         'Blood_glucose_level' : Blood_glucose_level
        
#     }

#     input_df = pd.DataFrame([raw_input])

#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0
            
#     input_df = input_df[expected_columns]
    
#     scaled_input = scaler.transform(input_df)
#     prediction = model.predict(scaled_input)[0]
    
#     if prediction ==1:
#         st.error("⚠️ High Risk of Diabetes")
#     else:
#         st.success("✅ Low Risk of Diabetes")




import streamlit as st
import joblib
import pandas as pd
import json
import requests
from streamlit_lottie import st_lottie

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="❤️", layout="wide")

# Custom CSS
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

# Doctor/Health Animation
lottie_health = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_tutvdkg0.json")

# -------------------- LOAD MODEL --------------------
model  = joblib.load("Random.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# -------------------- APP UI --------------------
st.title("🩺 Diabetes Prediction by Harsh ❤️")
st_lottie(lottie_health, height=200, key="health")

st.markdown("### 👉 Please provide the following details:")

col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 100, 40)
    Gender = st.selectbox("Gender", ['M','F'])
    Hypertension = st.selectbox("Hypertension", [0,1])
    Heart_disease = st.selectbox("Heart Disease", [0,1])

with col2:
    Smoking_history = st.selectbox("Smoking History", ['Never','No Info','Current','Former','Ever','Not Current'])
    Bmi = st.number_input("Body Mass Index", 10.0, 100.0, 20.0)
    HbA1c_level = st.number_input("Hemoglobin A1c test", 4.0, 16.0, 5.0)
    Blood_glucose_level = st.number_input("Blood Glucose Level", 50, 300, 120)

# -------------------- PREDICTION --------------------
if st.button('🔮 Predict'):
    raw_input = {
        'Age': Age,
        'Hypertension': Hypertension,
        'Heart_disease': Heart_disease,
        'Smoking_history': Smoking_history,
        'Bmi': Bmi,
        'HbA1c_level': HbA1c_level,
        'Blood_glucose_level': Blood_glucose_level
    }

    # Gender encoding (one-hot)
    raw_input['Gender_'+Gender] = 1

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error("🚨 **High Risk of Diabetes Detected!** ⚠️ Please consult a doctor immediately.")
        st_lottie(load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json"), height=200, key="alert")
    else:
        st.success("🎉 **Low Risk of Diabetes!** ✅ Stay healthy & keep maintaining your lifestyle.")
        st_lottie(load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_touohxv0.json"), height=200, key="success")
