import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict whether they will Pass or Fail.")

# User Inputs
age = st.slider("Age", 15, 22, 17)
studytime = st.selectbox("Weekly Study Time (hours)", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.slider("Absences", 0, 30, 5)
goout = st.slider("Going Out (social life, 1=low, 5=high)", 1, 5, 3)
freetime = st.slider("Free Time (1=low, 5=high)", 1, 5, 3)
Dalc = st.slider("Workday Alcohol Consumption (1=low, 5=high)", 1, 5, 1)
Walc = st.slider("Weekend Alcohol Consumption (1=low, 5=high)", 1, 5, 2)

# Convert to DataFrame
input_data = pd.DataFrame([[age, studytime, failures, absences, goout, freetime, Dalc, Walc]],
                          columns=['age', 'studytime', 'failures', 'absences', 'goout', 'freetime', 'Dalc', 'Walc'])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_input)[0]

# Show result
if st.button("Predict"):
    result = "✅ Pass" if prediction == 1 else "❌ Fail"
    st.subheader(f"Prediction: {result}")