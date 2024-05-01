# Import statements
import streamlit as st
import numpy as np
import pandas as pd
import tracemalloc
import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Memory allocation tracing
tracemalloc.start()

# Data preparation
text_data = "data/text.csv"

dataset = pd.read_csv(text_data)

tweets = dataset["text"].tolist()
categories = dataset["label"].tolist()

train_size = 300000
embedding_dim = 100
max_length = 128
trunc_type = "post"
pad_type = "post"

train_text = tweets[0:train_size]
test_text = tweets[train_size:]

# Text tokenizer
tokenizer = Tokenizer(num_words=48244, oov_token="<OOV>")
tokenizer.fit_on_texts(train_text)


# Process text for classification
def preprocess_text(text):
    text = text.lower()
    input_sequence = tokenizer.texts_to_sequences([text])
    input_padded = pad_sequences(input_sequence, maxlen=100, padding="post")

    return input_padded

@st.cache_resource
def model_preloading():
    classifier = load_model("text/text_emotion_model.h5")
    return classifier


text_emotion_model = model_preloading()

input_text = st.text_input("Type in sentence")
input = preprocess_text(input_text)

# Streamlit App UI
st.set_page_config(page_title="Feel_Net", page_icon=":tada:")

# Overview
st.title("Feel_Net")
st.subheader("Overview")
st.markdown(
    """ Included emotions => 'sadness , joy, love, anger, fear, surprise """
)

# Classifier function
score = text_emotion_model.predict(input)

emotions = ['sadness (0)', 'joy (1)', 'love (2)', 'anger (3)', 'fear (4)', 'surprise (5)']

predicted_class = np.argmax(score)
certainty = 100 * np.max(score)


try:
    if input_text:
        if st.button("Classify Text"):
            with st.spinner("Checking..."):
                st.markdown(
                    f"""
                    <div style="background-color: black; color: white; font-weight: bold; padding: 1rem; border-radius: 10px;">
                    <h4>Results</h4>
                        <h5>Input text => </h5>
                        <p>{input_text}</p>
                        <p>
                        Predicted emotion => <span style="font-weight: bold;">{emotions[predicted_class]}</span> with <span style="font-weight: bold;">{certainty:.2f}% </span>certainty
                        </p>
                    </div>
                        """,
                    unsafe_allow_html=True,
                )
                st.success("Successful")

except Exception as e:
    st.error(e)


st.write("")
st.write("")


st.markdown(
    "<hr style='border: 1px dashed #ddd; margin: 2rem;'>", unsafe_allow_html=True
)  # Horizontal line

st.markdown(
    """
    <div style="text-align: center; padding: 1rem;">
        Project by <a href="https://github.com/kelechi-c" target="_blank" style="color: white; font-weight: bold; text-decoration: none;">
         kelechi_tensor</a>
    </div>
    
    <div style="text-align: center; padding: 1rem;">
        Training data from <a href="https://kaggle.com" target="_blank" style="color: lightblue; font-weight: bold; text-decoration: none;">
         Kaggle</a>
    </div>
""",
    unsafe_allow_html=True,
)