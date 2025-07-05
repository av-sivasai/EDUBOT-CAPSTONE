import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------
# Load trained model and scaler
# --------------------------
model = pickle.load(open('attrition_rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("🚀 Employee Attrition Risk Prediction")
st.write("Upload your HR data and get attrition risk predictions to retain top talent proactively.")

uploaded_file = st.file_uploader("Upload HR CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(data.head())

    # Encoding categorical columns consistently
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'MaritalStatus', 'OverTime']
    le = pickle.load(open('label_encoder.pkl', 'rb')) if 'label_encoder.pkl' in locals() else None
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes

    # Scale numeric columns
    num_cols = ['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
                'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears',
                'YearsAtCompany']
    data[num_cols] = scaler.transform(data[num_cols])
    # Drop 'Attrition' column if present
    if 'Attrition' in data.columns:
        data = data.drop(columns=['Attrition'])

    # Predict
    predictions = model.predict(data)
    prediction_probs = model.predict_proba(data)[:,1]

    data['Attrition_Predicted'] = predictions
    data['Attrition_Probability'] = np.round(prediction_probs, 3)

    st.write("### Prediction Results")
    st.dataframe(data[['Attrition_Predicted', 'Attrition_Probability']].head())

    st.download_button("Download Predictions as CSV", data.to_csv(index=False).encode('utf-8'), "attrition_predictions.csv", "text/csv")

else:
    st.info("Please upload a CSV file to get started.")

st.write("---")
st.write("Built for your AI/ML Capstone to monitor attrition proactively.")
