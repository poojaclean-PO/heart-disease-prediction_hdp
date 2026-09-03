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
# Load Saved Models
# ============================================================

@st.cache_resource
def load_models():

    # Logistic Regression pipeline
    logistic_model = joblib.load(
        "heart_logistic_pipeline.pkl"
    )

    # Tuned Random Forest pipeline
    # This pipeline already contains preprocessing
    # + Random Forest model
    random_forest_model = joblib.load(
        "heart_tuned_rf_pipeline.pkl"
    )

    # Saved threshold information
    thresholds = joblib.load(
        "heart_model_thresholds.pkl"
    )

    return (
        logistic_model,
        random_forest_model,
        thresholds
    )


# Load models
logistic_model, random_forest_model, thresholds = load_models()


# ============================================================
# Model Thresholds
# ============================================================

# Logistic Regression threshold selected during
# previous model evaluation
LOGISTIC_THRESHOLD = 0.68


def get_rf_threshold(thresholds):

    """
    Get the tuned Random Forest threshold from
    heart_model_thresholds.pkl.

    If the saved file contains a Random Forest
    threshold, use it.

    Otherwise use the validated tuned threshold
    of 0.50.
    """

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(thresholds, dict):

        possible_keys = [
            "random_forest_tuned",
            "rf_threshold",
            "random_forest_threshold",
            "RF_THRESHOLD",
            "tuned_rf_threshold",
            "threshold"
        ]

        for key in possible_keys:

            if key in thresholds:

                value = thresholds[key]

                # Handle scalar values
                if isinstance(
                    value,
                    (int, float)
                ):

                    return float(value)

        # Check dictionary values for a numeric value
        for value in thresholds.values():

            if isinstance(
                value,
                (int, float)
            ):

                # We only use this fallback if
                # there is exactly one numeric value
                numeric_values = [
                    v for v in thresholds.values()
                    if isinstance(v, (int, float))
                ]

                if len(numeric_values) == 1:

                    return float(value)


    # --------------------------------------------------------
    # Single numeric threshold
    # --------------------------------------------------------

    if isinstance(
        thresholds,
        (int, float)
    ):

        return float(thresholds)


    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if isinstance(
        thresholds,
        pd.DataFrame
    ):

        # If a Model column exists, try to find RF row
        if "Model" in thresholds.columns:

            model_values = (
                thresholds["Model"]
                .astype(str)
                .str.lower()
            )

            rf_rows = thresholds[
                model_values.str.contains(
                    "random|rf",
                    regex=True,
                    na=False
                )
            ]

            if not rf_rows.empty:

                if "Threshold" in rf_rows.columns:

                    return float(
                        rf_rows["Threshold"].iloc[0]
                    )

        # Otherwise look for Threshold column
        if "Threshold" in thresholds.columns:

            return float(
                thresholds["Threshold"].iloc[0]
            )


    # --------------------------------------------------------
    # Validated tuned RF threshold
    # --------------------------------------------------------

    return 0.50


# Get Random Forest threshold
RF_THRESHOLD = get_rf_threshold(thresholds)


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
    # Tuned Random Forest Prediction
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