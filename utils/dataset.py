import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import csv
import random
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import time

img_mean = np.array([151.3254/255, 151.6072/255, 151.6273/255])   # mean value of nuPlan img
img_std = np.array([36.0859/255, 36.1013/255, 35.6058/255])     # std dev of nuPlan img

# Resize the camera frame by cropping from 200x300 to 200x200 then resizing to 224x224 for ResNet
def square_camera_image(image):
    transform = transforms.Compose([
        transforms.CenterCrop((200, 200)),  # Center crop the smaller edge (200x300 → 200x200)
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),      # Resize to 224x224
        transforms.ToTensor(),               # Convert to tensor
        transforms.Normalize(
          mean=img_mean,
          std=img_std
      )
    ])

    img = Image.fromarray(image)

    transformed_img = transform(img)

    return(transformed_img)

# Unnormalize camera image for plotting
def unnormalize_img(normalized_img:np.array):
    # normalized_img = normalized_img.numpy()  # if it's a tensor
    img_unorm = normalized_img * img_std[:, None, None] + img_mean[:, None, None]
    img_unorm = (255 * img_unorm.transpose(1, 2, 0).clip(0, 1)).astype(np.uint8)  # (H, W, C)
    return img_unorm

class DrivingDataset(Dataset):
    def __init__(self, file_list, test=False):
        self.samples = file_list
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.samples[idx], 'rb') as f:
            data = pickle.load(f)

        camera = data['camera']
        camera = square_camera_image(camera)

        history = torch.FloatTensor(data['sdc_history_feature'])

        if not self.test:
          future = torch.FloatTensor(data['sdc_future_feature'])
          return {
            'camera': camera,
            'history': history,
            'future': future
          }
        else:
          return {
            'camera': camera,
            'history': history
          }