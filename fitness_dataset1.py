
import pandas as pd
import streamlit as st
import joblib


# LOAD
model = joblib.load("fitness_dataset1.joblib")
model_columns = joblib.load("model_columns.joblib")
le_option = joblib.load("label_encoder_option.joblib")
le_training = joblib.load("label_encoder_training.joblib")
scaler = joblib.load("scaler.joblib")

st.title("FITNESS PREDICTOR ⚽")
st.write("Enter the player's details below...")


# USER INPUTS

Endurance = st.number_input("Endurance (meters)",)
speed = st.number_input("Speed Linear (seconds)",)
speed1 = st.number_input("Speed Max (seconds)",)
agility = st.number_input("Agility (seconds)",)
power = st.number_input("Lower Body Power (cm)",)
strength = st.number_input("Lower Body Strength",)
composition = st.number_input("Body Composition (%)",)

position = st.selectbox("Player Position", ("DF", "MF", "FW", "GK"))
training_status = st.selectbox("Training Status", ("Completed", "Partial", "Not Completed"))

# PREDICTION PIPELINE

if st.button("Predict Fitness"):

    # Encode categorical values
    position_encoded = le_option.transform([position])[0]
    training_encoded = le_training.transform([training_status])[0]

    # Prepare dataframe in correct order
    input_data = pd.DataFrame([[Endurance, speed, speed1, agility, power, strength, composition,
                                position_encoded, training_encoded]], 
                                columns = model_columns)

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"Player's Fitness Level: **{prediction}**")








