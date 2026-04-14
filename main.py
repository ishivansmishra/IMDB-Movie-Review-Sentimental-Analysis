import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import re

## Load the word IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## Load the pre-trained model with ReLU Activation
model = load_model('simple_rnn.imdb.h5')

# step-2 L Helper functions
# function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# function to preprocess the user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove punctuation
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

### Prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.45 else "Negative"

    return sentiment, prediction[0][0]



## streamlit ap

import streamlit as st
st.title('IMDB MOvie Review Sentimental Analysis')
st.write('Enter a movie review to classify it as postive or negative.')

## User Input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    ## Make prediction
    prediction = model.predict(preprocess_input)
    sentiment = "Positive" if prediction[0][0] < 0.45 else "Negative"

    ## Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction score: {prediction[0][0]}")


else:
    st.write('Please enter a movie review.')