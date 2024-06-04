import librosa
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from config import audio_config


# Audio
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()


def show_waveform(audio_file):
    waveform, sr = librosa.load(audio_file)
    librosa.display.waveshow(waveform, sr=sr)
    plt.title("Waveplot with Time Stretching", size=15)
    plt.show()


def get_audio_files_and_labels(dir):
    for folder,_, filenames in os.walk(dir):
        for file in filenames:
            file_path = os.path.join(folder, file)

            label = file.split("_")[-1].split(".")[0]

            yield (file_path, label)


def extract_audio_features(audio):
    audio_waveform, sr = librosa.load(
        audio, sr=audio_config.sample_rate, duration=3, offset=0.5
    )

    audio_mel = librosa.feature.mfcc(y=audio_waveform, sr=sr, n_mfcc=50)
    processed_mfcc = np.mean(audio_mel.T, axis=0)

    features = np.array(processed_mfcc)
    return features


def audio_file_csv(audio_files, labels):

    data_for_df = {
        "audio": audio_files,
        "emotion_labels": labels
    }

    feature_df = pd.DataFrame(data_for_df)
    feature_df.to_csv(audio_config.audio_feature_csv, index=False)

    print('Audio features csv creation complete')

    return feature_df


def load_audio_features(csv):
    df = pd.read_csv(csv)

    audio_feats = df["audio"]
    labels = df["emotion_labels"]
    
    return audio_feats, labels


# Images

def cv_readimg(image_file):
    image = cv2.imread(image_file)
    # image = cv2.cvtColor(image, )
    image = cv2.resize(image, (200, 200))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image

    return image
