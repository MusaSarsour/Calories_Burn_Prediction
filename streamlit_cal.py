import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('cal_model.pkl')

# Load the scaler
scaler = joblib.load('cal_scaler.pkl')

# Title
st.title('Calories Burned Prediction')

# Age
age = st.number_input('Age', min_value=20, max_value=79, value=20)

# Height
height = st.number_input('Height (cm)', min_value=123, max_value=222, value=123)

# Weight
weight = st.number_input('Weight (kg)', min_value=36, max_value=132, value=36)

# Duration
duration = st.number_input('Duration (minutes)', min_value=1, max_value=30, value=1)

# Heart rate
heart_rate = st.number_input('Heart rate', min_value=67, max_value=128, value=67)

# body temperature
body_temp = st.number_input('Body temperature', min_value=37.0, max_value=42.0, value=37.0, step=0.1)

# Gender
gender = st.radio("Gender", ["Male", "Female"])

# Encoding: Male -> [0,1], Female -> [1,0]
Gender = [0, 1] if gender == "Male" else [1, 0]

# Layout: Buttons side by side
col1, col2, col3, col4, col5 = st.columns(5)

# Reset Button
with col1:
    reset_button = st.button("Reset 🔄")

with col2:
    # Prediction
    if st.button('Predict 🥵'):
        # Create a numpy array
        input_data = np.array([age, height, weight, duration, heart_rate, body_temp, Gender[0], Gender[1]]).reshape(1, -1)
        # Scale the input
        input_data = scaler.transform(input_data)
        # Predict
        prediction = model.predict(input_data)
        # Display the result
        st.write(f'You burned {prediction[0]:.2f} calories 🏃🔥')
