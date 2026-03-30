# Streamlit 示例代码
import streamlit as st

st.title("Streamlit 数据可视化")
st.write("以下是一个简单的数据表格：")

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

st.table(data)