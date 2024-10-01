import torch
import numpy as np
import os

from torch.utils.data import DataLoader

from neuralnet import CricketShotClassifier

model = CricketShotClassifier()

model.load_state_dict(torch.load("cricket_shot_classifier.pth"))

model.eval()

with torch.no_grad():
    for inputs in 