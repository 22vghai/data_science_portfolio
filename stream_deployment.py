import streamlit as st
import numpy as np 
import pickle 

model = pickle.load(open('model.pkl', 'rb'))

st.title('California House Price Predictor')

# Input sliders 
# Add sliders for all 8 inputs 

MedInc = st.slider('Median Income', 1.0, 15.0, 5.0)
HouseAge = st.slider('House Age', 1, 50, 25)
AveRooms = st.slider('Average Rooms', 1.0, 10.0, 5.0)
AveBedrms = st.slider('Average Bedrooms', 0.5, 5.0, 1.0)
Population = st.slider('Population', 100, 4000, 1000)
AveOccup = st.slider('Average Occupancy', 1.0, 10.0, 3.0)
Latitude = st.slider('Latitude', 32.0, 42.0, 37.0)
Longitude = st.slider('Longitude', -124.0, -114.0, -119.0)
  

# Put them into the model in the same order 
features  = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

pred = model.predict(features)

st.subheader(f"Predicted Median House Value: ${pred[0]*100000:.2f}")

