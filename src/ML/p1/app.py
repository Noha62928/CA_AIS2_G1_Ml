import streamlit as st 
st.title("Boston House Price Prediction")
st.write("Welcome!")
name = st.text_input("What is your name?") 
if st.button("Submit"): 
    st.write("Hello", name)
