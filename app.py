from pathlib import Path

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# Model File Paths
# ============================================================

# Get the folder where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Final Logistic Regression model and threshold
LR_MODEL_PATH = BASE_DIR / "heart_logistic_final_pipeline.pkl"
LR_THRESHOLD_PATH = BASE_DIR / "heart_logistic_final_threshold.pkl"

# Final Random Forest model and threshold
RF_MODEL_PATH = BASE_DIR / "heart_rf_final_pipeline.pkl"
RF_THRESHOLD_PATH = BASE_DIR / "heart_rf_final_threshold.pkl"


# ============================================================
# Load Final Saved Models
# ============================================================

@st.cache_resource
def load_models():

    # Load final Logistic Regression pipeline
    logistic_model = joblib.load(LR_MODEL_PATH)

    # Load final Logistic Regression threshold
    logistic_threshold = float(
        joblib.load(LR_THRESHOLD_PATH)
    )

    # Load final Random Forest pipeline
    random_forest_model = joblib.load(RF_MODEL_PATH)

    # Load final Random Forest threshold
    random_forest_threshold = float(
        joblib.load(RF_THRESHOLD_PATH)
    )

    return (
        logistic_model,
        logistic_threshold,
        random_forest_model,
        random_forest_threshold
    )


# Load models
(
    logistic_model,
    LOGISTIC_THRESHOLD,
    random_forest_model,
    RF_THRESHOLD
) = load_models()


# ============================================================
# Application Title
# ============================================================

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's information below and click "
    "**Predict**."
)


# ============================================================
# Patient Input Form
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# Column 1
# ============================================================

with col1:

    BMI = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

    Smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )

    AlcoholDrinking = st.selectbox(
        "Alcohol Drinking",
        ["No", "Yes"]
    )

    Stroke = st.selectbox(
        "Stroke",
        ["No", "Yes"]
    )

    PhysicalHealth = st.number_input(
        "Physical Health",
        min_value=0.0,
        max_value=30.0,
        value=0.0,
        step=1.0
    )

    MentalHealth = st.number_input(
        "Mental Health",
        min_value=0.0,
        max_value=30.0,
        value=0.0,
        step=1.0
    )

    DiffWalking = st.selectbox(
        "Difficulty Walking",
        ["No", "Yes"]
    )

    Sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    AgeCategory = st.selectbox(
        "Age Category",
        [
            "18-24",
            "25-29",
            "30-34",
            "35-39",
            "40-44",
            "45-49",
            "50-54",
            "55-59",
            "60-64",
            "65-69",
            "70-74",
            "75-79",
            "80 or older"
        ]
    )


# ============================================================
# Column 2
# ============================================================

with col2:

    Race = st.selectbox(
        "Race",
        [
            "White",
            "Black",
            "Asian",
            "American Indian/Alaskan Native",
            "Hispanic",
            "Other"
        ]
    )

    Diabetic = st.selectbox(
        "Diabetic",
        [
            "No",
            "Yes",
            "No, borderline diabetes",
            "Yes (during pregnancy)"
        ]
    )

    PhysicalActivity = st.selectbox(
        "Physical Activity",
        ["No", "Yes"]
    )

    GenHealth = st.selectbox(
        "General Health",
        [
            "Excellent",
            "Very good",
            "Good",
            "Fair",
            "Poor"
        ]
    )

    SleepTime = st.number_input(
        "Sleep Time (hours)",
        min_value=1.0,
        max_value=24.0,
        value=7.0,
        step=1.0
    )

    Asthma = st.selectbox(
        "Asthma",
        ["No", "Yes"]
    )

    KidneyDisease = st.selectbox(
        "Kidney Disease",
        ["No", "Yes"]
    )

    SkinCancer = st.selectbox(
        "Skin Cancer",
        ["No", "Yes"]
    )


# ============================================================
# Prediction Button
# ============================================================

if st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
):

    # ========================================================
    # Create Input DataFrame
    # ========================================================

    input_data = pd.DataFrame({

        "BMI": [BMI],

        "Smoking": [Smoking],

        "AlcoholDrinking": [AlcoholDrinking],

        "Stroke": [Stroke],

        "PhysicalHealth": [PhysicalHealth],

        "MentalHealth": [MentalHealth],

        "DiffWalking": [DiffWalking],

        "Sex": [Sex],

        "AgeCategory": [AgeCategory],

        "Race": [Race],

        "Diabetic": [Diabetic],

        "PhysicalActivity": [PhysicalActivity],

        "GenHealth": [GenHealth],

        "SleepTime": [SleepTime],

        "Asthma": [Asthma],

        "KidneyDisease": [KidneyDisease],

        "SkinCancer": [SkinCancer]
    })


    # ========================================================
    # Logistic Regression Prediction
    # ========================================================

    logistic_probability = (
        logistic_model.predict_proba(
            input_data
        )[0][1]
    )

    logistic_prediction = int(
        logistic_probability >= LOGISTIC_THRESHOLD
    )


    # ========================================================
    # Random Forest Prediction
    # ========================================================

    rf_probability = (
        random_forest_model.predict_proba(
            input_data
        )[0][1]
    )

    rf_prediction = int(
        rf_probability >= RF_THRESHOLD
    )


    # ========================================================
    # Prediction Results
    # ========================================================

    st.divider()

    st.subheader("Prediction Results")

    result_col1, result_col2 = st.columns(2)


    # ========================================================
    # Logistic Regression Result
    # ========================================================

    with result_col1:

        st.markdown(
            "### Logistic Regression"
        )

        st.metric(
            "Heart Disease Probability",
            f"{logistic_probability * 100:.2f}%"
        )

        st.write(
            f"Threshold: **{LOGISTIC_THRESHOLD:.2f}**"
        )

        if logistic_prediction == 1:

            st.error(
                "⚠️ Prediction: Heart Disease"
            )

        else:

            st.success(
                "✅ Prediction: No Heart Disease"
            )


    # ========================================================
    # Random Forest Result
    # ========================================================

    with result_col2:

        st.markdown(
            "### Random Forest"
        )

        st.metric(
            "Heart Disease Probability",
            f"{rf_probability * 100:.2f}%"
        )

        st.write(
            f"Threshold: **{RF_THRESHOLD:.2f}**"
        )

        if rf_prediction == 1:

            st.error(
                "⚠️ Prediction: Heart Disease"
            )

        else:

            st.success(
                "✅ Prediction: No Heart Disease"
            )


    # ========================================================
    # Show Patient Input
    # ========================================================

    with st.expander(
        "View Patient Input"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )