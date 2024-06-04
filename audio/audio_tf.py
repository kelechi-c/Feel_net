import tensorflow as tf
import keras
from keras import layers as tf_nn
from keras import models
from config import audio_config, Configs
from audio_data2 import x_audio_train, x_audio_test, y_labels_test, y_labels_train

epochs = Configs.epochs

audio_clf = models.Sequential(
    [
        tf_nn.LSTM(128, return_sequences=True, input_shape=(50, 1)),
        tf_nn.Dense(64, activation='relu'),
        tf_nn.Dropout(0.2),
        tf_nn.Dense(16, activation='relu'),
        tf_nn.Dropout(0.2),
        tf_nn.Dense(audio_config.num_classes, activation='softmax')
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=Configs.lr)
rlp_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, verbose=0, patience=5, min_lr=0.00001) # type: ignore

audio_clf.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

audio_clf.summary()