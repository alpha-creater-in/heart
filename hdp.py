import streamlit as st
import joblib
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------- HEADER --------------------
st.title("🫀 Heart Disease Prediction")


st.divider()

# -------------------- INPUT SECTION --------------------
st.subheader(" Medical Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 1, 120)

    sex_option = st.selectbox(
        "Sex",
        ["Male", "Female", "Prefer not to say"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina (0)",
         "Atypical Angina (1)",
         "Non-anginal Pain (2)",
         "Asymptomatic (3)"]
    )

    bp = st.number_input("Resting Blood Pressure (mm Hg)")
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)")

    fbs_option = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

with col2:
    ekg = st.selectbox(
        "Resting ECG Results",
        ["Normal (0)",
         "ST-T Wave Abnormality (1)",
         "Left Ventricular Hypertrophy (2)"]
    )

    max_hr = st.number_input("Maximum Heart Rate Achieved")

    exercise_angina = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    st_depression = st.number_input("ST Depression (Oldpeak)", step=0.1)

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping (0)",
         "Flat (1)",
         "Downsloping (2)"]
    )

    vessels = st.selectbox(
        "Number of Major Vessels Colored by Fluoroscopy",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thallium Stress Test Result",
        ["Normal (0)",
         "Fixed Defect (1)",
         "Reversible Defect (2)",
         "Other (3)"]
    )

# -------------------- VALUE MAPPING --------------------

# Sex mapping
if sex_option == "Male":
    sex = 1
elif sex_option == "Female":
    sex = 0
else:
    sex = 0   # default encoding (since model trained only 0/1)

# Chest pain mapping
cp = ["Typical Angina (0)",
      "Atypical Angina (1)",
      "Non-anginal Pain (2)",
      "Asymptomatic (3)"].index(chest_pain)

# FBS mapping
fbs = 1 if fbs_option == "Yes" else 0

# ECG mapping
ekg_value = ["Normal (0)",
             "ST-T Wave Abnormality (1)",
             "Left Ventricular Hypertrophy (2)"].index(ekg)

# Exercise angina mapping
ex_angina = 1 if exercise_angina == "Yes" else 0

# Slope mapping
slope_value = ["Upsloping (0)",
               "Flat (1)",
               "Downsloping (2)"].index(slope)

# Thal mapping
thal_value = ["Normal (0)",
              "Fixed Defect (1)",
              "Reversible Defect (2)",
              "Other (3)"].index(thal)

st.divider()

# -------------------- PREDICTION --------------------
if st.button("🔍 Assess Risk"):

    input_data = np.array([[age, sex, cp, bp, cholesterol, fbs,
                            ekg_value, max_hr, ex_angina,
                            st_depression, slope_value,
                            vessels, thal_value]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Assessment Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nEstimated Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nEstimated Probability: {probability*100:.2f}%")

