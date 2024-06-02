import librosa
import torchaudio
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import Configs

audio_folder = ''

# Audio

def mel_spectrogram(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
    
    plt.figure()
    plt.imshow(spectrogram.log2()[0,:,:].numpy(), cmap='viridis')
    

# Images

def cv_readimg(image_file):
    image = cv2.imread(image_file)
    image = cv2.COLOR_BGR2RGB(image)
    image = cv2.resize(image, (200, 200))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize the image

    return image