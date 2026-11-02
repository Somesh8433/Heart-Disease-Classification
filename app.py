
import streamlit as st
import pandas as pd
import joblib
import json

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Classification",
    page_icon="❤️",
    layout="wide"
)

# Load model and feature information
model = joblib.load("heart_disease_model.pkl")

with open("feature_schema.json", "r") as f:
    schema = json.load(f)

st.title("❤️ Heart Disease Risk Classification")

st.markdown("""
### Machine Learning Based Prediction System

Enter the patient information below to obtain a
heart disease classification prediction.
""")

st.warning(
    "Educational prototype only. This prediction is not "
    "a medical diagnosis or a substitute for professional care."
)

st.divider()

# Patient information form
st.subheader("Patient Information")

with st.form("prediction_form"):

    input_data = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(schema["feature_columns"]):

        container = col1 if i % 2 == 0 else col2

        with container:

            if feature in schema["categorical_features"]:

                options = schema["category_options"][feature]

                input_data[feature] = st.selectbox(
                    feature,
                    options=options
                )

            else:

                stats = schema["numeric_stats"][feature]

                input_data[feature] = st.number_input(
                    feature,
                    min_value=stats["min"],
                    max_value=stats["max"],
                    value=stats["median"]
                )

    submitted = st.form_submit_button(
        "Predict Heart Disease Risk"
    )

# Prediction
if submitted:

    input_df = pd.DataFrame(
        [input_data],
        columns=schema["feature_columns"]
    )

    prediction = model.predict(input_df)[0]

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "The model predicts the positive heart disease class."
        )

    else:

        st.success(
            "The model predicts the negative heart disease class."
        )

    # Probability
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        positive_index = classes.index(1) if 1 in classes else -1

        probability = probabilities[positive_index]

        st.metric(
            "Model-estimated positive-class probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.caption(
            "This is a model probability, not a clinically "
            "validated estimate of an individual's actual risk."
        )

st.divider()

st.caption(
    "Heart Disease Risk Classification | Machine Learning Project"
)
