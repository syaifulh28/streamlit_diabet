import streamlit as st
import pandas as pd
import joblib
import gdown

# Function to download model from Google Drive
@st.cache_resource
def download_model_from_gdrive():
    url = "https://drive.google.com/file/d/1teGSB1uWQsMbmE_Y2oxsl50jkPbeoXQO/view?usp=sharing"  # Replace 'YOUR_FILE_ID' with the file ID from Google Drive
    output = "model.pkl"
    gdown.download(url, output, quiet=False)
    return output

# Function to load the model
@st.cache_resource
def load_model():
    model_path = download_model_from_gdrive()  # Download model if not already present
    model = joblib.load(model_path)
    return model

# Initialize the app
st.title("Classification Model Deployment")
st.write("Upload a CSV file for prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(data)

    # Load the model
    model = load_model()

    # Perform predictions
    if st.button("Predict"):
        predictions = model.predict(data)
        st.write("Predictions:")
        data["Prediction"] = predictions
        st.dataframe(data)

        # Option to download the result
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
