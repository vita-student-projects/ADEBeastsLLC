import torch
from torch.utils.data import DataLoader
import os

from utils.dataset import DrivingDataset
from utils.model import DrivingPlanner
from utils.train import validate
from utils.evaluate import visualize_with_depth, evaluate_model

val_data_dir = "val"
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]
val_dataset = DrivingDataset(val_files)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

# Loach parameters
use_depth = True
use_semantic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load testing model
model = DrivingPlanner(use_depth_aux=True, use_semantic_aux=True)
# model.load_state_dict(torch.load("results/slurm-2636515(2).pt"))
model.load_state_dict(torch.load("Weights_V2.pt"))
model.to(device)

######################################## Evaluate ########################################
# 🔚 Call at the end after training both models
visualize_with_depth(val_loader, model, device=device)

evaluate_model(model, val_loader, device)