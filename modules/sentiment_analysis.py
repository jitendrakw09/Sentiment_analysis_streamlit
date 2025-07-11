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
        st.warning("⚠ Please upload a chat log first.")
        return

    df = pd.read_csv(chat_path)
    sid = SentimentIntensityAnalyzer()
    df["sentiment"] = df["message"].apply(lambda x: sid.polarity_scores(str(x))["compound"])
    df["label"] = df["sentiment"].apply(lambda s: "Positive" if s > 0.2 else ("Negative" if s < -0.2 else "Neutral"))
    df.to_csv("data/analyzed_chat.csv", index=False)

    st.success("✅ Sentiment analysis complete!")
    st.dataframe(df.head())

    if st.checkbox("🔓 Use Gemini for Enhanced Sentiment"):
        api_key = st.text_input("Enter your Gemini API key", type="password")
        if api_key and st.button("Run Gemini Analysis"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")

                sentiments = []
                for msg in df["message"]:
                    prompt = f"Classify the sentiment of this message as Positive, Neutral, or Negative:\n\n'{msg}'"
                    response = model.generate_content(prompt)
                    label = response.text.strip()
                    sentiments.append(label)

                df["gemini_label"] = sentiments
                df.to_csv("data/analyzed_chat.csv", index=False, encoding="utf-8")
                df.to_csv("data/analyzed_chat_gemini.csv", index=False, encoding="utf-8")
                st.success("✅ Gemini sentiment analysis complete!")
                st.dataframe(df.head())
            except ImportError:
                st.error("Gemini library not installed. Run pip install google-generativeai.")
            except Exception as e:
                st.error(f"Gemini Error: {str(e)}")
