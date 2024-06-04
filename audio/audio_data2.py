import librosa
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from util_funcs import get_audio_files_and_labels, extract_audio_features, audio_file_csv
from config import audio_config

emotion_dict = {
    'hap': 'happy',
    'dis': 'disgusted',
    'ang': 'angry'
}

# For Audio paths: GEt file paths, labels, then map the extract audio function

file_paths, labels = zip(*get_audio_files_and_labels(audio_config.audio_folder))

audio_df = audio_file_csv(file_paths, labels)

# Extract audio features and store in numpy array
audio_features = audio_df["audio"].apply(lambda v: extract_audio_features(v)) # type: ignore 

x_audio = [x for x in audio_features]
x_audio = np.array(x_audio)
x_audio = np.expand_dims(x_audio, -1)

print(x_audio.shape)


# For the labels: replace with complete text, and encode as numbers
encoder = OneHotEncoder()

y_labels = encoder.fit_transform(audio_df[["emotion_labels"]])
y_labels = np.array(y_labels)

