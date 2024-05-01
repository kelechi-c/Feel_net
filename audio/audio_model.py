import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchaudio
import torchsummary as summary


class AudioModel(nn.Module):
    def __init__(self):
        super(AudioModel, self).__init__()
        self.layer1 = nn.Sequential([
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])
        
        self.layer2 = nn.Sequential([
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])
        
        self.layer3 = nn.Sequential([
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])

        self.layer4 = nn.Sequential([
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        ])
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128*5*4, 128)
        self.linear2 = nn.Linear(128, 7)
        self.output = nn.Softmax(dim=1)
        
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        logits = self.linear2(x)
        
        return self.output(logits)
    

audio_classifier = AudioModel()
summary(audio_classifier, (1, 28, 28))
        