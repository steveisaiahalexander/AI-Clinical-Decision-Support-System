import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Disease Predictor", layout="wide")
st.title("🩺 Disease Symptom Predictor")
st.markdown("### Enter your symptoms to get disease predictions")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("outputs/final_disease_model.pkl")
    le = joblib.load("outputs/label_encoder.pkl")
    symptoms = joblib.load("outputs/symptom_columns.pkl")
    return model, le, symptoms

model, le, symptom_cols = load_model()

# Multi-select symptoms
selected_symptoms = st.multiselect(
    "Select all symptoms you are experiencing:",
    options=symptom_cols,
    help="You can select multiple symptoms"
)

if st.button("🔍 Predict Disease", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        input_vec = pd.Series(0, index=symptom_cols)
        for sym in selected_symptoms:
            input_vec[sym] = 1
        
        proba = model.predict_proba([input_vec.values])[0]
        top_idx = proba.argsort()[::-1][:5]
        
        st.success("### Top 5 Predicted Diseases")
        for i in top_idx:
            disease = le.classes_[i]
            confidence = proba[i] * 100
            st.metric(label=disease, value=f"{confidence:.1f}%")

        st.info("Note: This is a machine learning prediction for educational purposes only. Consult a doctor for medical advice.")

# Footer
st.caption("Built with Random Forest | Trained on Kaggle Disease Symptom Dataset")