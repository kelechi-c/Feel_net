import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import os
from config import model_config

dataset_folder = "/kaggle/input/audio-emotions/Emotions"
data_config = model_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionDataset(Dataset):
    def __init__(self, folder_path, transform, device, target_sample_rate, num_samples):
        self.folder_path = folder_path
        self.device = device
        self.transform = transform.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.file_paths = []
        self.labels = []

        for label_dir in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label_dir)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    file_path = os.path.join(label_path, file_name)
                    self.file_paths.append(file_path)
                    self.labels.append(label_dir)

    def __len__(self):
        return len(self.file_paths)
    
    def _resample(self, waveform, sample_rate):
        resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
        return resampler(waveform)
        
    def _mix_down(self, waveform):
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform 
    
    def _cut(self, waveform):
        if waveform.shape[1]>self.num_samples:
            waveform = waveform[:, :self.num_samples]
        
        return waveform
    
    def _right_pad(self, waveform):
        signal_length = waveform.shape[1]
        if signal_length < self.num_samples:
            num_padding = self.num_samples - signal_length
            waveform = torch.nn.functional.pad(waveform, (0, num_padding))
        
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(self.device)
        waveform = self._resample(waveform, sample_rate)
        waveform = self._mix_down(waveform)
        waveform = self.cut(waveform)
        waveform = self._right_pad(waveform)
        waveform = self.transform(waveform)

        return waveform, float(label)


melspectogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=data_config['sample_rate'], n_fft=1024, hop_length=512, n_mels=64
)

audio_dataset = EmotionDataset(dataset_folder, melspectogram, device, data_config['sample_rate'], data_config['num_samples'])

train_size = int(0.8 * len(audio_dataset))
valid_size = len(audio_dataset) - train_size

train_dataset, valid_dataset = random_split(audio_dataset, [train_size, valid_size])

train_loader = DataLoader(
    train_dataset, batch_size=data_config["batch_size"], shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=data_config["batch_size"], shuffle=False
)
