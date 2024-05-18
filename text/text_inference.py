from tensorflow import keras as tfkeras
from keras import layers, losses
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from data_prep import texts


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

test_sentences = [
    "I definitely love machine learning, its so cool",
    "Twitter is a really difficult place, so much hate",
    "I am okay, not so bad",
    "Oh I am overflowing with joy and happiness",
    'What is all this nonsense? I am so frustrated',
    'That was really scary, my knees '
]

# sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)


def preprocess_text(text):
    text = text.lower()
    input_sequence = tokenizer.texts_to_sequences([text])
    input_padded = pad_sequences(input_sequence, maxlen=100, padding="post")

    return input_padded
