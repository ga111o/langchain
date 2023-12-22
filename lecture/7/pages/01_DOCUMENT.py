import streamlit as st
import time

st.set_page_config(
    page_title= "DOCUMENT",
    page_icon="📄",
)

st.title("ga111o! DOCUMENT")

with st.chat_message("human"):
    st.write("ga11o!")

with st.chat_message("ai"):
    st.write("how are u?")

st.chat_input("SEND A MESSAGE")

with st.status("embeding file..", expanded=True) as status:
    time.sleep(2)
    st.write("getting the file")
    time.sleep(3)
    st.write("embeding the file")
    time.sleep(3)
    st.write("caching the file")
    status.update(label="ERROR", state="error")

