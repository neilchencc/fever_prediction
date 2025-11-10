import streamlit as st

st.title("你好，Streamlit Cloud 👋")
name = st.text_input("请输入你的名字：")
if st.button("提交"):
    st.success(f"你好，{name}！欢迎来到 Streamlit Cloud！")