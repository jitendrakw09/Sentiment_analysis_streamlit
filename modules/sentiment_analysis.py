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
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=10
                    )
                    label = response.choices[0].message.content.strip()
                    sentiments.append(label)

                df["gpt_label"] = sentiments
                df.to_csv("data/analyzed_chat_gpt.csv", index=False)
                st.success("✅ GPT sentiment analysis complete!")
                st.dataframe(df.head())
            except ImportError:
                st.error("OpenAI library not installed. Run `pip install openai`.")
            except Exception as e:
                st.error(f"OpenAI Error: {str(e)}")
