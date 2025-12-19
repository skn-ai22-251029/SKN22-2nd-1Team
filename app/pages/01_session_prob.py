# app/pages/02_what_if.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="01_session_prob", layout="wide")

st.title("01_session_prob 시뮬레이터")
st.write("여기가 01_session_prob 페이지입니다.")
