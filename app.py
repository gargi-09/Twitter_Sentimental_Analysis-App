# core packages
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

# EDA packages
import plotly.express as pex

# utils
import joblib

pipe_lr = joblib.load(open("emotion_detection_1.pkl", "rb"))


def predict_emotions(doc):
    results = pipe_lr.predict([doc])
    return results


def get_prediction_proba(doc):
    results = pipe_lr.predict_proba([doc])
    return results


emotion_dict = {"happy": "😄", "sadness": "😢", "love": "😍", "anger": "😠", "fear": "😨😱",
                "surprise-curiosity": "😮🤔", "neutral": "😐", "hate": "😡", "gratitude": "🤗","excitement" : "🥳"}


def main():
    st.title("Sentiment Analysis App")

    menu = ["Home", "Search by Emotion", "About"]
    with st.sidebar:

        choice = option_menu("Main Menu", menu,
                             icons=['house', 'emoji-sunglasses', 'info-circle'], menu_icon="cast", default_index=1)

    if choice == "Home":
        st.subheader("Home-Emotion In Text")

        with st.form(key="Emotion_clf_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            with col1:
                st.success("Orignal Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write(f"{prediction[0]} : {emotion_dict[prediction[0]]}")
                st.write(f"Confidence : {round(np.max(probability),2)}")

            with col2:
                st.success("Prediction probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['emotions', 'probability']

                fig = pex.bar(proba_df_clean, x='emotions',
                              y='probability', color='emotions')
                st.plotly_chart(fig, use_container_width=True)

    elif choice == "Search by Emotion":
        st.subheader("Search by a specific emotion")
        df = pd.read_csv('output.csv')
        fig1 = pex.histogram(df, x="Emotion", color="Emotion")
        st.plotly_chart(fig1, use_container_width=True)

        em = st.selectbox("Emotion", emotion_dict.keys())

        emotion = df[df['Emotion'] == em]
        st.dataframe(emotion)

    else:
        st.subheader("About the App")
        st.markdown("Welcome to our **:red[Sentiment analysis]** web app! We help you analyze text/tweets and gauge the sentiment behind them. Our app uses advanced machine learning algorithms to accurately identify and classify tweets as different emotions or sentiments. Try it out now and see what Twitter users are saying or just how you feel about literally anything!")

if __name__ == '__main__':
    main()
