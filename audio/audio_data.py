import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
import os

dataset_folder = "/kaggle/input/audio-emotions/Emotions"


class EmotionDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
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

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label
    
audio_dataset = EmotionDataset(dataset_folder)

train_size = int(0.8 * len(audio_dataset))
valid_size = len(audio_dataset) - train_size

train_dataset, valid_dataset = random_split(audio_dataset, [train_size, valid_size])
