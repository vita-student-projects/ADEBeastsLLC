import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from utils.dataset import DrivingDataset
from utils.logger import Logger
from utils.model import DrivingPlanner, FirstModel
from utils.train import train, validate
from datetime import datetime

# Format: MM-DD_HH-MM-SS
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
filename = f"output_{timestamp}"
model_save_path = filename + ".pt"

train_data_dir = "train"
val_data_dir = "val"

train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]

train_dataset = DrivingDataset(train_files)
val_dataset = DrivingDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

######################################## Training Parameters ########################################

use_depth = True
use_semantic = True

# Load previous weights (old architecture)
pretrained_model = FirstModel()
pretrained_model.load_state_dict(torch.load("Weights_V1.pth"))

# # Load new model
model = DrivingPlanner(use_depth_aux=True, use_semantic_aux=True)

# Load pre-trined weight on to new model
model.history_encoder.load_state_dict(pretrained_model.history.state_dict())
model.future_decoder.load_state_dict(pretrained_model.decoder.state_dict())

tot_epochs = 80
logger = Logger(tot_epochs)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tot_epochs, eta_min=1e-4)

train(model, train_loader, val_loader, optimizer,
      logger, num_epochs=tot_epochs, start_epoch=0,
      lambda_depth=0.75, lambda_semantic=1.75,
      use_depth_aux=use_depth, use_semantic_aux=use_semantic,
      scheduler=scheduler)

torch.save(model.save_dict(), model_save_path)