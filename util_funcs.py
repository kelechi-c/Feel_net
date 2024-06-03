import librosa
import torchaudio
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Configs, audio_config

audio_folder = ''


# Audio
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    figure, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    figure.suptitle(title)
    figure.tight_layout()


def get_mel_spectrogram(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)

    return spectrogram


def show_waveform(audio_file):
    waveform, sr = librosa.loadO(audio_file)
    librosa.display.waveshow(waveform, sr)
    plt.title("Waveplot with Time Stretching", size=15)
    plt.show()


def get_audio_files_and_labels(dir):
    for folder, filenames in os.walk(dir):
        for file in filenames:
            file_path = os.path.join(folder, file)

            label = file.split("_")[-1].split(".")[0]

            yield (file_path, label)


def extract_audio_features(audio):
    audio_waveform, sr = librosa.load(audio, sr=audio_config.sample_rate)
    
    audio_mfcc = librosa.feature.melspectrogram(audio_waveform, sr, n_mfcc=50)
    processed_mfcc = np.mean(audio_mfcc.T, axis=0)
    
    features = np.array(processed_mfcc)
    
    return features

def audio_featre_csv(audio_files, labels):
    audio_features = []
    emotions = []
    
    for audio, label in zip(audio_files, labels):
       audio_feature = extract_audio_features(audio)
       emotion = label.lower()
       
       audio_features.append(audio_feature)
       emotions.append(emotion)
       
    return audio_features, emotions


# def cv_read_spectrogram(melspec):


# Images

def cv_readimg(image_file):
    image = cv2.imread(image_file)
    image = cv2.COLOR_BGR2RGB(image)
    image = cv2.resize(image, (200, 200))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image

    return image
