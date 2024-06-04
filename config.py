class Configs:
    # General
    epochs = 30
    lr = 0.01
    batch_size = 32
    

class audio_config:
    audio_folder = '/kaggle/input/audio-sentiment-analysis/Audio_Dataset'
    audio_feature_csv = 'audio_emotions.csv'
    sample_rate = 22000
    cmap = 'inferno'
    
    
class image_config:
    image_size = 200