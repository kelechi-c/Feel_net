class Configs:
    # General
    epochs = 35
    lr = 0.001
    batch_size = 32
    

class audio_config:
    audio_folder = '/kaggle/input/audio-sentiment-analysis/Audio_Dataset'
    audio_feature_csv = 'audio_emotions.csv'
    sample_rate = 22000
    cmap = 'inferno'
    num_classes = 6
    