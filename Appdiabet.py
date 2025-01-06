import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# Debugging utility
def log_debug(message):
    st.write(f"DEBUG: {message}")

# Function to download model from Google Drive
@st.cache_resource
def download_model_from_gdrive():
    try:
        url = "https://drive.google.com/file/d/1teGSB1uWQsMbmE_Y2oxsl50jkPbeoXQO/view?usp=sharing"  # Ganti 'YOUR_FILE_ID' dengan ID file Google Drive
        output = "model.pkl"
        log_debug("Downloading model from Google Drive...")
        
        # Download the file
        gdown.download(url, output, quiet=False)
        log_debug(f"Model downloaded successfully. File exists: {os.path.exists(output)}")
        
        return output
    except Exception as e:
        log_debug(f"Error during model download: {e}")
        raise

# Function to load the model
@st.cache_resource
def load_model():
    try:
        log_debug("Starting to load model...")
        
        # Download and load the model
        model_path = download_model_from_gdrive()
        log_debug(f"Model file path: {model_path}")
        
        model = joblib.load(model_path)
        log_debug("Model loaded successfully.")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        log_debug(f"Error loading model: {e}")
        raise

# Initialize the app
st.title("Classification Model Deployment")
st.write("Upload a CSV file for prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)
        
        # Load the model
        log_debug("Attempting to load the model...")
        model = load_model()

        # Perform predictions
        if st.button("Predict"):
            log_debug("Performing prediction...")
            predictions = model.predict(data)
            log_debug(f"Predictions: {predictions}")

            st.write("Predictions:")
            data["Prediction"] = predictions
            st.dataframe(data)

            # Option to download the result
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        log_debug(f"Error in main execution: {e}")
