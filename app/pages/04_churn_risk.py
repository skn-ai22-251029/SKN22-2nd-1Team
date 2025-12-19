# app/pages/02_what_if.py
import streamlit as st
from ui.header import render_header

render_header()

st.set_page_config(page_title="churn_risk", layout="wide")


st.title("churn_risk")
st.write("여기가 churn_risk 페이지입니다.")
