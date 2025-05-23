import torch
from torch.utils.data import DataLoader
import os

from utils.dataset import DrivingDataset
from utils.model import DrivingPlanner_GRU
from utils.evaluate import visualize, evaluate_model
from utils.train import train_params, customCriterion

val_data_dir = "val_real"
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]
val_dataset = DrivingDataset(val_files[500:])
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

# Load testing model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrivingPlanner_GRU()
model.load_state_dict(torch.load("results/ADE_1_4270.pt"))
model.to(device)

# Initialize parameters
eval_params = train_params()
eval_params.set_device(device)
eval_params.set_val_loader(val_loader)
eval_params.set_criterion(
	customCriterion(
      	x_scale = 1.0,
      	y_scale = 1.0,
      	heading_scale = 4.0
	)
)

######################################## Evaluate ########################################
# 🔚 Call at the end after training both models
visualize(val_loader, model, device=device)

evaluate_model(model, eval_params)