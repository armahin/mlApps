import streamlit as st
import numpy as np
import xgboost as xgb
import pickle
import math
import os

# Load the trained XGBoost model
def load_model():
    with open('C:/Users/User/Desktop/Project/trainedmodel.sav','rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions
def predict_embodied_carbon(model, inputs):
    data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(data)
    carbon = 236.5 * math.e ** (6.606 * 10 ** (-3) * prediction)
    return carbon

# Function to display the app
def main():
    st.title("Low Carbon Mix Design")

    st.write("Please input 5 sets of Mix Designs:")

    inputs = []
    for i in range(5):
        st.subheader(f"MIX DESIGN {i+1}")
        features = []
        for j, feature_name in enumerate(["Cement", "GGBS", "Fly Ash", "Silica Fume", "WB Ratio"]):
            key = f"input_{i}_{feature_name}"
            feature = st.number_input(feature_name, key=key, value=0.0, step=0.1)
            features.append(feature)
        inputs.append(features)

    inputs = np.array(inputs)

    if st.button("Generate Predictions"):
        model = load_model()
        predictions = []
        for input_data in inputs:
            prediction = predict_embodied_carbon(model, input_data)
            predictions.append(prediction)

        median_index = np.argsort(predictions)[len(predictions)//2]  # Get index of median value
        st.write(f"<b>Cement, GGBS, Fly Ash, Sillica Fume, WB Ratio</b>: {inputs[median_index]}",unsafe_allow_html=True)
port = int(os.environ.get('PORT', 8501))

# Run the Streamlit app
if __name__ == '__main__':
    st.port = port
    main()
