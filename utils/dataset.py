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
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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

# resizing the depth image from 200x300 to 224x224
def square_depth_image(depth_image):
    # Step 1: Crop center 200x200 region from width
    # depth_image shape: (200, 300, 1)
    cropped = depth_image[:, 50:250, :]  # Crop width: take center (300 → 200)

    # Step 2: Convert to torch tensor with shape (1, 1, H, W)
    depth_tensor = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).float()

    # Step 3: Resize to (224, 224) using bilinear interpolation
    depth_resized = F.interpolate(depth_tensor, size=(224, 224), mode='bilinear', align_corners=False)

    # Step 4: Remove batch/channel dimensions → return (224, 224)

    # print(depth_resized.shape)
    return depth_resized.squeeze(0)

# resizing the semantic map from 200x300 to 224x224
def square_semantic_map(semantic_map):
  # semantic_map is a 200x300x3 RGB image
  transform = transforms.Compose([
      transforms.CenterCrop((200, 200)),
      transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
  ])
  # print("semantic map shape: ", semantic_map.shape)
  # Input must be a PIL image or Tensor in (C, H, W)
  resized_map = transform(semantic_map)         # Output: (3, 224, 224)
  return resized_map

# Unnormalize camera image for plotting 
def unnormalize_img(normalized_img:np.array, mean:np.array, std:np.array):
  normalized_img = normalized_img.numpy()  # if it's a tensor
  img_unorm = normalized_img * std[:, None, None] + mean[:, None, None]
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
        depth = data['depth'] / 255.0 # values 0-254
        # print("og depth shape: ", depth.shape)

        depth = square_depth_image(depth)
        # print("new dpeth chape: ", depth.shape)
        semantic_map = torch.FloatTensor(data['semantic_label']) # values: 0-14

        # Semantic labels
        semantic_colormap = {
            0: [0, 0, 0],         # UNLABELED
            1: [0, 0, 142],       # CAR
            2: [0, 0, 70],        # TRUCK
            3: [220, 20, 60],     # PEDESTRIAN
            4: [119, 11, 32],     # BIKE
            5: [152, 251, 152],   # TERRAIN
            6: [128, 64, 128],    # ROAD
            7: [244, 35, 232],    # SIDEWALK
            8: [70, 130, 180],    # SKY
            9: [250, 170, 30],    # TRAFFIC_LIGHT
            10: [190, 153, 153],  # FENCE
            11: [220, 220, 0],    # TRAFFIC_SIGN
            12: [255, 255, 255],  # LANE_LINE
            13: [55, 176, 189],   # CROSSWALK
            14: [0, 60, 100]      # BUS
        }

        height, width = semantic_map.shape
        rgb_semantic_img = torch.zeros((3, height, width), dtype=torch.float32)
        for label, color in semantic_colormap.items():
            for i in range(3):
                rgb_semantic_img[i][semantic_map == label] = color[i] / 255.0

        # print("og semantic map shape: ", semantic_map.shape)
        semantic_map = square_semantic_map(rgb_semantic_img)
        # print("new semantic map shape: ", semantic_map.shape)

        if not self.test:
            future = torch.FloatTensor(data['sdc_future_feature'])
            return {
                'camera': camera,
                'history': history,
                'depth': depth,
                'future': future,
                'semantic_label': semantic_map
            }
        else:
            return {
                'camera': camera,
                'history': history,
                'depth': depth
            }
