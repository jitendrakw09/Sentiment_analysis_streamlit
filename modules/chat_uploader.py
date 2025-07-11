# 📁 modules/chat_uploader.py

def upload_page():
    import streamlit as st
    import pandas as pd
    import os

    st.header("📤 Upload Chat Log")
    st.markdown("Upload a CSV file with columns: `timestamp`, `user`, `message`")

    uploaded_file = st.file_uploader("Upload chat CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in ["timestamp", "user", "message"]):
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/chat.csv", index=False)
            st.success("✅ Chat uploaded successfully!")
            st.dataframe(df.head())
        else:
            st.error("❌ Missing required columns. Please include timestamp, user, and message.")
