import os
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils.dataset import DrivingDataset
from utils.logger import Logger
from utils.model import DrivingPlanner, DrivingPlanner_GRU
from utils.train import train, validate, customCriterion, train_params
from utils.evaluate import visualize
from datetime import datetime

# Format: MM-DD_HH-MM-SS
train_data_dir = "train"
val_data_dir = "val_real"

train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.pkl')]
val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.pkl')]

split_num = 500
train_files_mixed = train_files + val_files[:split_num]
val_files = val_files[split_num:]

train_dataset = DrivingDataset(train_files_mixed)
val_dataset = DrivingDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

######################################## Training Parameters ########################################

# Load new model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrivingPlanner_GRU()
tot_epochs = 250

# Prepare training parameters
training_params = train_params()
training_params.set_tot_epochs(tot_epochs)
training_params.set_device(device)
training_params.set_logger(Logger(training_params.tot_epochs))
training_params.set_train_loader(train_loader)
training_params.set_val_loader(val_loader)
training_params.set_criterion(
	customCriterion(
      	x_scale = 2.0,
      	y_scale = 2.0,
      	heading_scale = 8.0
	)
)
training_params.set_optimizer(
    optim.Adam(
		model.parameters(),
		lr=8e-4,
		weight_decay=1e-5
    )
)
training_params.set_scheduler(
    optim.lr_scheduler.CosineAnnealingLR(
		training_params.optimizer,
		T_max=training_params.tot_epochs,
		eta_min=5e-5
	)
)
training_params.set_data_aug_T(
	# Define available transformations
    [
		None, None,
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        T.RandomPerspective(distortion_scale=0.5, p=1.0),
        T.RandomResizedCrop(size=(224, 224)),
        T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
		T.ElasticTransform(),
		T.RandomAffine(degrees=(0,0), translate=(0.25,0), shear=2)
    ]
)

train(model,training_params)

######################################## Evaluate ########################################
# 🔚 Call at the end after training both models
visualize(val_loader, model, training_params.device)