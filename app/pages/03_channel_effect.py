# app/pages/02_what_if.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="channel_effect", layout="wide")


st.title("channel_effect")
st.write("여기가 channel_effect 페이지입니다.")
