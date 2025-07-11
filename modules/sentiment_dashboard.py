# 📁 modules/sentiment_dashboard.py

def dashboard_page():
    import streamlit as st
    import pandas as pd
    import os
    import plotly.express as px

    st.header("📊 Sentiment Dashboard")

    csv_path = "data/analyzed_chat.csv"
    if not os.path.exists(csv_path):
        st.warning("⚠️ Run sentiment analysis first.")
        return

    df = pd.read_csv(csv_path)

    st.subheader("Sentiment Distribution")
    sentiment_counts = df["label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    fig = px.pie(sentiment_counts, names="Sentiment", values="Count", title="Overall Sentiment")
    st.plotly_chart(fig)

    st.subheader("User Sentiment Trends")
    if "user" in df.columns:
        user_df = df.groupby(["user", "label"]).size().reset_index(name="count")
        fig2 = px.bar(user_df, x="user", y="count", color="label", barmode="group", title="Sentiment by User")
        st.plotly_chart(fig2)

    st.subheader("Sentiment Over Time")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df_time = df.dropna(subset=["timestamp"])
    if not df_time.empty:
        timeline = df_time.groupby([df_time["timestamp"].dt.date, "label"]).size().reset_index(name="count")
        fig3 = px.line(timeline, x="timestamp", y="count", color="label", markers=True, title="Sentiment Timeline")
        st.plotly_chart(fig3)
