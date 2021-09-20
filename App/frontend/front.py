import requests
import streamlit as st
import requests

st.title("Corona tweet sentiment analysis app")

st.write("Takes tweet about corona virus cleans it analyse it and gives output positive, negative or nutral as a output")

tweet=st.text_area("Paste a tweet")

if st.button("Predict"):
    if tweet is not None:
        res=requests.post(f"http://127.0.0.1:8000/predict/{tweet}")
        resp=res.json()
        pred=resp.get("prediction")
        st.write(pred)