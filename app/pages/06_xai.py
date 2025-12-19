# app/pages/02_what_if.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="xai", layout="wide")


st.title("xai 시뮬레이터")
st.write("여기가 xai 페이지입니다.")
