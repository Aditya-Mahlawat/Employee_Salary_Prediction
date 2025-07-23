import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Page configuration
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

# App title and description
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Improvement 1: Added caching for model loading
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Please run the retrain_model.py script first to create a compatible model.")
    model_loaded = False

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education_num = st.sidebar.selectbox("Education Level", [
    ("HS-grad", 9),
    ("Some-college", 10),
    ("Bachelors", 13),
    ("Masters", 14),
    ("PhD", 16),
    ("Assoc", 12),
    ("11th", 7),
    ("10th", 6),
    ("7th-8th", 4),
    ("Prof-school", 15),
    ("9th", 5),
    ("12th", 8),
    ("Doctorate", 16)
], format_func=lambda x: x[0])

occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces", "Others"
])

workclass = st.sidebar.selectbox("Work Class", [
    "Private", "Self-emp-not-inc", "Local-gov", "State-gov",
    "Self-emp-inc", "Federal-gov", "Others"
])

marital_status = st.sidebar.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])

hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

# Build input DataFrame with all required features
input_data = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [200000],  # Default value
    'educational-num': [education_num[1]],  # Use the numeric value
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': ["Not-in-family"],  # Default value
    'race': ["White"],  # Default value
    'gender': ["Male"],  # Default value
    'capital-gain': [0],  # Default value
    'capital-loss': [0],  # Default value
    'hours-per-week': [hours_per_week],
    'native-country': ["United-States"]  # Default value
}

input_df = pd.DataFrame(input_data)

# Display simplified input data to the user
display_df = pd.DataFrame({
    'age': [age],
    'education': [education_num[0]],  # Display the text value
    'occupation': [occupation],
    'workclass': [workclass],
    'marital-status': [marital_status],
    'hours-per-week': [hours_per_week]
})

st.write("### 🔎 Input Data")
st.write(display_df)

# Function to preprocess data like in the training notebook
def preprocess_data(df):
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Apply label encoding to categorical columns
    categorical_cols = ['workclass', 'marital-status', 'occupation', 
                       'relationship', 'race', 'gender', 'native-country']
    
    encoder = LabelEncoder()
    for col in categorical_cols:
        df_copy[col] = encoder.fit_transform(df_copy[col])
    
    # Apply scaling
    scaler = MinMaxScaler()
    return scaler.fit_transform(df_copy)

# Improvement 2: Added loading spinner during prediction
if st.button("Predict Salary Class") and model_loaded:
    with st.spinner("Predicting..."):
        try:
            # Preprocess the input data
            processed_input = preprocess_data(input_df)
            
            # Make prediction
            prediction = model.predict(processed_input)
            st.success(f"✅ Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check that your input data matches the format expected by the model.")

# Batch prediction section
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")

# Improvement 3: Added helpful caption for better user guidance
st.caption("Upload a CSV file with columns matching the required model features.")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

# Improvement 4: Added error handling for batch prediction
if uploaded_file is not None and model_loaded:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())
        
        # Check if we can map the columns to required features
        required_features = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
                           'occupation', 'relationship', 'race', 'gender', 'capital-gain',
                           'capital-loss', 'hours-per-week', 'native-country']
        
        # If education is provided but not educational-num, map it
        if 'education' in batch_data.columns and 'educational-num' not in batch_data.columns:
            education_map = {
                "HS-grad": 9, "Some-college": 10, "Bachelors": 13,
                "Masters": 14, "PhD": 16, "Assoc": 12, "11th": 7,
                "10th": 6, "7th-8th": 4, "Prof-school": 15, "9th": 5,
                "12th": 8, "Doctorate": 16
            }
            batch_data['educational-num'] = batch_data['education'].map(education_map).fillna(10)
        
        # Fill in missing columns with defaults
        for feature in required_features:
            if feature not in batch_data.columns:
                if feature == 'workclass':
                    batch_data[feature] = 'Private'
                elif feature == 'fnlwgt':
                    batch_data[feature] = 200000
                elif feature == 'marital-status':
                    batch_data[feature] = 'Never-married'
                elif feature == 'relationship':
                    batch_data[feature] = 'Not-in-family'
                elif feature == 'race':
                    batch_data[feature] = 'White'
                elif feature == 'gender':
                    batch_data[feature] = 'Male'
                elif feature == 'capital-gain' or feature == 'capital-loss':
                    batch_data[feature] = 0
                elif feature == 'native-country':
                    batch_data[feature] = 'United-States'
        
        # Ensure we have all required columns in the right order
        batch_data_processed = batch_data[required_features]
        
        # Preprocess the batch data
        processed_batch = preprocess_data(batch_data_processed)
        
        # Make predictions
        batch_preds = model.predict(processed_batch)
        
        # Add predictions to original data
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Make sure your CSV file has the necessary columns or can be mapped to the required features.")

