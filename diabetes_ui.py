import streamlit as st
import joblib
import pandas as pd


model  = joblib.load("Random.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")




st.title("Diabetes prediction by harsh ❤️")
st.markdown("Provide the following details")
Age = st.slider("Age",18,100,40)
Gender = st.selectbox("SEX",['M','F'])
Hypertension = st.selectbox("Hyper Tension",[0,1])
Heart_disease = st.selectbox("Heart Disease",[0,1])
Smoking_history = st.selectbox("Smoking History",['Never','No Info','Current','Former','Ever','Not Current'])
Bmi = st.number_input("Body Mass Index",10.0,100.0,20.0)
HbA1c_level = st.number_input("Hemoglobin A1c test",4.0,16.0,5.0)
Blood_glucose_level = st.number_input("Blood Glucose Level",50,300,120)




if st.button('Predict'):
    raw_input = {
        'Age': Age,
        'Gender'+ Gender: 1,
        'Hypertension': Hypertension,
        'Heart_disease' :Heart_disease,
        'Smoking_history' : Smoking_history,
        'Bmi' : Bmi,
        'HbA1c_level' : HbA1c_level,
        'Blood_glucose_level' : Blood_glucose_level
        
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if prediction ==1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")