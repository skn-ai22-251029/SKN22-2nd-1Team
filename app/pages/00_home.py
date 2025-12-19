# app/app.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="My Streamlit App", layout="wide")


st.title("홈")
st.write("여기가 홈 페이지입니다.")
