import streamlit as st
import pickle

import string


# from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
email_model = pickle.load(open('Email_Model.pkl', 'rb'))



st.title("Email/SMS/News Spam Classifier")

input_sms = st.text_area("Enter the message")

col1, col2 = st.columns([.5,1])
with col1:
    if st.button('Predict Email'):

    # 1. preprocess
        transformed_sms = transform_text(input_sms)
    # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
    # 3. predict
        result = email_model.predict(vector_input)[0]
    # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
with col2:
    if st.button('Predict News'):
        st.header("Feature is under Process")
