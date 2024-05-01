import torch
from tqdm import tqdm
from config import model_config
from audio_data import train_loader, valid_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

