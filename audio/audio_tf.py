import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras import layers as tf_nn
from keras import models

from config import audio_config, Configs
from audio_data2 import x_audio_train, x_audio_test, y_labels_test, y_labels_train, input_shape
from util_funcs import accuracy_plot, loss_plot


epochs = Configs.epochs

audio_clf = models.Sequential(
    [
        tf_nn.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf_nn.Dense(64, activation="relu"),
        tf_nn.Dropout(0.2),
        tf_nn.Dense(16, activation="relu"),
        tf_nn.Dropout(0.2),
        tf_nn.Dense(8, activation="relu"),
        tf_nn.Dropout(0.2),
        tf_nn.Dense(6, activation="softmax"),
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=Configs.lr)
rlp_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, verbose=0, patience=5, min_lr=0.00001) # type: ignore

audio_clf.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

audio_clf.summary()

training = audio_clf.fit(
    x_audio_train,
    y_labels_train,
    validation_split=0.1,
    epochs=Configs.epochs,
    batch_size=Configs.batch_size,
    callbacks=[rlp_callback],
)

epochs = list(range(Configs.epochs))

loss = training.history["loss"]
val_loss = training.history["val_loss"]

accuracy = training.history["accuracy"]
val_accuracy = training.history["val_accuracy"]

accuracy_plot(epochs, accuracy, val_accuracy)
loss_plot(epochs, loss, val_loss)
