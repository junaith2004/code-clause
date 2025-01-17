import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    # Using sample dataset from sklearn
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
    data = pd.DataFrame(X, columns=columns)
    data['target'] = y
    return data

def train_model(data):
    # Prepare features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Streamlit UI
def main():
    st.title("Heart Disease Risk Prediction")
    st.write("Enter your health metrics to assess your heart disease risk")
    
    # Create input form
    with st.form("health_metrics"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, value=40)
            gender = st.selectbox("Gender", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type", 
                ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 
                min_value=90, max_value=200, value=120)
            cholesterol = st.number_input("Cholesterol Level (mg/dl)", 
                min_value=100, max_value=600, value=200)
            
        with col2:
            blood_sugar = st.number_input("Fasting Blood Sugar (mg/dl)", 
                min_value=70, max_value=400, value=100)
            ecg = st.selectbox("Resting ECG Results", 
                ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
            max_heart_rate = st.number_input("Maximum Heart Rate", 
                min_value=60, max_value=220, value=150)
            exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
            st_depression = st.number_input("ST Depression Induced by Exercise", 
                min_value=0.0, max_value=6.0, value=0.0)
            
        submit_button = st.form_submit_button("Predict Risk")
    
    if submit_button:
        # Transform inputs into model features
        features = pd.DataFrame({
            'age': [age],
            'sex': [1 if gender == "Male" else 0],
            'cp': [["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain)],
            'trestbps': [blood_pressure],
            'chol': [cholesterol],
            'fbs': [1 if blood_sugar > 120 else 0],
            'restecg': [["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(ecg)],
            'thalach': [max_heart_rate],
            'exang': [1 if exercise_angina == "Yes" else 0],
            'oldpeak': [st_depression]
        })
        
        try:
            # Load data and train model (in practice, you'd want to save/load the trained model)
            data = load_data()
            model, scaler = train_model(data)
            
            # Scale features and make prediction
            features_scaled = scaler.transform(features)
            prediction = model.predict_proba(features_scaled)[0]
            
            # Display results
            st.subheader("Risk Assessment Results")
            risk_percentage = prediction[1] * 100
            
            # Create a progress bar for risk visualization
            st.progress(risk_percentage / 100)
            
            if risk_percentage < 20:
                risk_level = "Low"
                color = "green"
            elif risk_percentage < 50:
                risk_level = "Moderate"
                color = "yellow"
            else:
                risk_level = "High"
                color = "red"
                
            st.markdown(f"**Risk Level:** <span style='color:{color}'>{risk_level}</span>", 
                unsafe_allow_html=True)
            st.write(f"Estimated risk percentage: {risk_percentage:.1f}%")
            
            # Add disclaimer
            st.warning("""
                This is a preliminary assessment tool and should not be used as a substitute 
                for professional medical advice. Please consult with a healthcare provider 
                for proper diagnosis and treatment.
                """)
            
        except Exception as e:
            st.error("An error occurred during prediction. Please ensure all inputs are valid.")
            st.error(str(e))

if __name__ == "__main__":
    main()
