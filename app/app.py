# app/app.py
import streamlit as st

# Streamlit 페이지 설정은 반드시 switch_page 이전
st.set_page_config(
    page_title="🚀SkN22-2nd-1Team",
    layout="wide"
)

# 앱 실행 시 홈 페이지로 즉시 이동
st.switch_page("pages/00_home.py")
