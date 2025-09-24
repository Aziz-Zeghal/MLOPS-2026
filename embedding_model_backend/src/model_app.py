import streamlit as st
import joblib

model = joblib.load("regression.joblib")

size = st.number_input("Size", min_value=0)
bedrooms = st.number_input("Number of Bedrooms", min_value=0)
garden = st.number_input("Has Garden? (1 for Yes, 0 for No)", min_value=0, max_value=1)

features = [[size, bedrooms, garden]]
prediction = model.predict(features)[0]
st.metric("Predicted Price", prediction)
