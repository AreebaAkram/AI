import streamlit as st
from PIL import Image
st.title("First UI")
st.write("sum of 2 numbers")
n1 = st.number_input("Enter first number: ")
n2 = st.number_input("Enter 2nd number : ")
if st.button("Calculate"):
    st.write(n1+n1)

salary = st.slider("select range" , min_value = 100, max_value = 500, value=(150, 250))
st.write(salary[0], salary[1])

pic = st.file_uploader("choose image", type=["jpg","png","jpeg"])
if pic is not None:
    img = Image.open(pic)
    st.image (img, caption= "logo")