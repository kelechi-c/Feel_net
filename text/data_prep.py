import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow import keras as tfkeras
from keras import layers, losses
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


text_data = "/kaggle/input/emotions/text.csv"

dataset = pd.read_csv(text_data)

texts = dataset["text"].tolist()
categories = dataset["label"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

vocab_size = len(tokenizer.word_index) + 1
print(f"Vocab size is {vocab_size}")


train_size = 300000
embedding_dim = 100
max_length = 128
trunc_type = "post"
pad_type = "post"

train_text = texts[0:train_size]
test_text = texts[train_size:]

train_labels = categories[0:train_size]
test_labels = categories[train_size:]

train_sequence = tokenizer.texts_to_sequences(train_text)
train_padded = pad_sequences(
    train_sequence, padding="post", maxlen=max_length, truncating=trunc_type
)

test_sequence = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(
    test_sequence, padding="post", maxlen=max_length, truncating=trunc_type
)

train_padded = np.array(train_padded)
train_labels = np.array(train_labels)
test_padded = np.array(test_padded)
test_labels = np.array(test_labels)

train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

test_labels = test_labels[: len(test_padded)]

dataset_shape = dataset.shape

train_labels = to_categorical(train_labels, num_classes=6)
test_labels = to_categorical(test_labels, num_classes=6)

