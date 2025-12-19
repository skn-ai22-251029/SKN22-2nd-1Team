# app/pages/02_what_if.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="What-if 시뮬레이터", layout="wide")

st.title("What-if 시뮬레이터")
st.write("여기가 02_what_if 페이지입니다.")
