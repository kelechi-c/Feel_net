import torch
import torch.nn as nn 
from torch.nn import functional as nn_func
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available is True else "cpu")

class AudioConvnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv2_block = nn.Sequential(
            nn.Conv2d(32, 64),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        
        self.flatten_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(51136, 50),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(50, 7)
        
    def forward(self, image):
        x = self.conv_block(image)
        x = self.conv2_block(x)
        x = self.flatten_linear(x)
        x = nn_func.dropout(x, training=self.training)
        x = nn_func.relu(self.linear(x))
        cn_output = nn_func.log_softmax(x, dim=1)
        
        return cn_output

audio_classifier = AudioConvnet().to(device=device)

# class AudioModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
#             nn.BatchNorm2d(),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )

#         self.layer2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )

#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )

#         self.layer4 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.flatten = nn.Flatten()
#         self.linear1 = nn.Linear(128 * 5 * 4, 128)
#         self.linear2 = nn.Linear(128, 7)
#         self.output = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.flatten(x)
#         x = self.linear1(x)
#         logits = self.linear2(x)

#         return self.output(logits)


# audio_classifier = AudioModel().cuda()

# model_summary = summary(audio_classifier, (1, 28, 28))

# print(model_summary)
