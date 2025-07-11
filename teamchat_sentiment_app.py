# 📁 teamchat_sentiment_analyzer/app.py

import streamlit as st
from modules import chat_uploader, sentiment_analysis, sentiment_dashboard

st.set_page_config(page_title="💬 TeamChat Sentiment Analyzer", layout="wide")
st.title("💬 TeamChat Sentiment Analyzer")

menu = st.sidebar.radio("Choose Feature", ["Upload Chat Log", "Analyze Sentiment", "View Dashboard"])

if menu == "Upload Chat Log":
    chat_uploader.upload_page()
elif menu == "Analyze Sentiment":
    sentiment_analysis.analyze_page()
elif menu == "View Dashboard":
    sentiment_dashboard.dashboard_page()


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
            df.to_csv("data/chat.csv", index=False)
            st.success("✅ Chat uploaded successfully!")
            st.dataframe(df.head())
        else:
            st.error("❌ Missing required columns. Please include timestamp, user, and message.")


# 📁 modules/sentiment_analysis.py

def analyze_page():
    import streamlit as st
    import pandas as pd
    import os
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon')

    st.header("🧠 Sentiment Analysis")

    chat_path = "data/chat.csv"
    if not os.path.exists(chat_path):
        st.warning("⚠️ Please upload a chat log first.")
        return

    df = pd.read_csv(chat_path)
    sid = SentimentIntensityAnalyzer()
    df["sentiment"] = df["message"].apply(lambda x: sid.polarity_scores(str(x))["compound"])
    df["label"] = df["sentiment"].apply(lambda s: "Positive" if s > 0.2 else ("Negative" if s < -0.2 else "Neutral"))
    df.to_csv("data/analyzed_chat.csv", index=False)

    st.success("✅ Sentiment analysis complete!")
    st.dataframe(df.head())

    if st.checkbox("🔓 Use OpenAI for Enhanced Sentiment"):
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        if api_key and st.button("Run GPT Analysis"):
            try:
                import openai
                openai.api_key = api_key

                sentiments = []
                for msg in df["message"]:
                    prompt = f"Classify the sentiment of this message as Positive, Neutral, or Negative:\n\n'{msg}'"
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    sentiments.append(response["choices"][0]["message"]["content"].strip())

                df["gpt_label"] = sentiments
                df.to_csv("data/analyzed_chat_gpt.csv", index=False)
                st.success("✅ GPT sentiment analysis complete!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"OpenAI Error: {str(e)}")


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
