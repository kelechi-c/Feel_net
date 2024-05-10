import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from config import model_config
from audio_data import train_loader, valid_loader
from audio_model import audio_classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = model_config()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(audio_classifier.parameters(), lr=config['lr'])
def training_step(model, train_loader, loss_fn, optimizer, device):
    acc_list = []

    for waveform, label in tqdm(train_loader):
        
        model.train()

        waveform = waveform.cuda()
        label = label.view(-1)
        label = label.cuda()

        logits = model(waveform)
        loss = loss_fn(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = label.size(0)
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == label).sum().item()
        acc_list.append(correct / total)

    print(f"Accuracy: {np.mean(acc_list)}, Loss: {loss.item()}")

def training_loop(model, train_loader, loss_fn, optimizer, device, epochs=config['epochs']):
    for epoch in tqdm(range(config['epochs'])):
        print(f'Training Epoch {epoch+1} of {epochs}....')
        training_step(model, train_loader, loss_fn, optimizer, device)
        print(f"Training of epoch {epoch+1} complete ✅")
        print('____________________')
    print("Training Complete ✅✅✅")


training_loop(audio_classifier, train_loader, loss_fn, optimizer, device)
