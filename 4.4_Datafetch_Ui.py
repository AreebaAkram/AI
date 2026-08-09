import requests
import streamlit as st
from bs4 import BeautifulSoup 
# from textblob import TextBlob
st.title("Data Fetch")
text = st.text_area("Enter your link")
if st.button("check"):
    response = requests.get(text)
    content = response.text
    blob = BeautifulSoup(content, 'html.parser')
    t1= st.text_area("Enter tag you want to find")
if st.button("enter"):
    data = blob.find_all(t1)
    for i in data:
        print(i.get_text())