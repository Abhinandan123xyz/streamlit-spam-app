import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
import re

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text]
    return " ".join(text)

st.title("📩 SMS Spam Classification App")
st.write("Enter a message and the model will detect whether it is **spam** or **not spam**.")

input_sms = st.text_area("Type your message here...")

if st.button("Predict"):
    cleaned_sms = preprocess_text(input_sms)
    vectorized = vectorizer.transform([cleaned_sms]).toarray()
    result = model.predict(vectorized)[0]

    if result == 1:
        st.error("🚨 This message is flagged as **SPAM**")
    else:
        st.success("✅ This message is **NOT SPAM**")
