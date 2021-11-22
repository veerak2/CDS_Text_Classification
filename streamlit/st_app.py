import requests
import streamlit as st

test = {}

test['transcript'] = st.text_input("Enter transcript")

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict",  json=test)

    st.text(response.text)