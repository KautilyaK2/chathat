import streamlit as st
prompt = st.chat_input("Say something")
if prompt:
    st.markdown(f"User has sent the following prompt: {prompt}")
