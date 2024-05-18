import tensorflow as tf
from tensorflow import keras as tfkeras
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras as tfkeras
from keras import layers
from config import Config
from data_prep import train_labels, test_labels, train_padded, test_padded


early_stop = tfkeras.callbacks.EarlyStopping(
    min_delta=0.001, patience=10, restore_best_weights=True
)

vocab_size = Config.vocab_size
embedding_dim = Config.embedding_dim

text_model = tfkeras.Sequential(
    [
        layers.Embedding(vocab_size, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(6, activation="softmax"),
    ]
)


text_model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)


history = text_model.fit(
    train_padded,
    train_labels,
    epochs=30,
    validation_data=(test_padded, test_labels),
    verbose=1,
    callbacks=[early_stop],
)


history_plot = pd.DataFrame(history.history)
history_plot.loc[:, ["loss", "val_loss"]].plot()
print("Minimum validation loss: {}".format(history_plot["val_loss"].min()))

results = text_model.evaluate(test_padded, test_labels, batch_size=32)

print(f"Your model has an accuracy of {100*results[1]}%")