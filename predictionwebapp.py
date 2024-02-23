# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 22:30:55 2024

@author: User
"""

import numpy as np
import pickle
import streamlit as st
import math

#loading the saved model
loaded_model = pickle.load(open('C:/Users/User/Desktop/Project/trainedmodel.sav','rb'))

#Creating the function for prediction

def cs_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)  # Your input data here
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # Get predictions from the model
    prediction = loaded_model.predict(input_data_reshaped)
    result = "Embodied Carbon is " + str(236.5 * math.e ** (6.606 * 10 ** (-3) * prediction))

    # Interpret the output based on your task
    return result


def main():
    
    # Title Giving 
    st.title('Embodied Carbon Prediction')
    
    # Getting the input from the user
    Cement = st.text_input('Cement')
    GGBS = st.text_input('GGBS')
    FlyAsh = st.text_input('Fly Ash')
    SilicaFume = st.text_input('Silica Fume')
    WBRatio = st.text_input('W/B Ratio')
    
    # Convert input to float if not empty, otherwise use default value
    Cement = float(Cement) if Cement else 0.0
    GGBS = float(GGBS) if GGBS else 0.0
    FlyAsh = float(FlyAsh) if FlyAsh else 0.0
    SilicaFume = float(SilicaFume) if SilicaFume else 0.0
    WBRatio = float(WBRatio) if WBRatio else 0.0
    
    # Convert input data into a NumPy array
    input_data = np.array([Cement, GGBS, FlyAsh, SilicaFume, WBRatio])
    
    # Code for prediction
    diagnosis = ''
    
    # Creating the button for prediction
    if st.button("Prediction Result"):
        diagnosis = cs_prediction(input_data)
        
    st.success(diagnosis)

if __name__ == '__main__':
    main()

    