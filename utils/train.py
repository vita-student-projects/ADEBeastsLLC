import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from utils.logger import Logger

class customCriterion(nn.Module):
	def __init__(self, x_scale = 1.0, y_scale = 1.0, heading_scale = 4.0):
		super(customCriterion, self).__init__()
		self.x_scale = x_scale
		self.y_scale = y_scale
		self.heading_scale = heading_scale

	def forward(self, predictions, target):
		x_error = predictions[:, :, 0] - target[:, :, 0]
		y_error = predictions[:, :, 1] - target[:, :, 1]
		heading_error = predictions[:, :, 2] - target[:, :, 2]
		loss = (self.x_scale * x_error**2 +
				self.y_scale * y_error**2 +
				self.heading_scale * heading_error**2).mean()
		return loss

# Class to pass all needed arguments for training
class train_params:
    def __init__(self):
        self.logger = None
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.tot_epochs = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.data_aug_T = None
        self.current_epoch = 0

        self.save_best = False
        self.save_best_fname = ""

    def set_logger(self, logger:Logger):
        self.logger = logger

    def set_device(self, device):
        self.device = device

    def set_train_loader(self, train_loader):
        self.train_loader = train_loader

    def set_val_loader(self, val_loader):
        self.val_loader = val_loader

    def set_tot_epochs(self, tot_epochs):
        self.tot_epochs = tot_epochs

    def set_save_best(self, save_name):
        self.save_best = True
        self.save_best_fname = save_name

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion: customCriterion):
        self.criterion = criterion

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_data_aug_T(self, transformation):
        self.data_aug_T = transformation

    def increase_epoch(self):
        self.current_epoch += 1

########################## TRAINING FUNCTIONS #######################
def train_one_epoch(model,
                    params: train_params):
                    
    model.train()
    train_loss = 0.0

    for batch in params.train_loader:
        cam, hist, fut = [batch[k].to(params.device) for k in ['camera', 'history', 'future']]

        # Choose random transformation
        transform = random.choice(params.data_aug_T)

        if transform is not None:
            # print("transform is: ", transform)
            # Apply data augmentation to each image in batch individually
            cam_transformed = []
            for img in cam:  
                # img_pil = T.ToPILImage()(img.cpu())     # Convert to PIL
                img_aug = transform(img)            # Apply dt augmentation
                # img_tensor = T.ToTensor()(img_aug).to(params.device)
                cam_transformed.append(img_aug)
            cam = torch.stack(cam_transformed)

        params.optimizer.zero_grad()
        fut_pred = model(cam, hist)
        
        loss = params.criterion(fut_pred, fut)
        loss.backward()
        params.optimizer.step()
        train_loss += loss.item()

    params.logger.log(
        epoch=params.current_epoch,
		loss_train=train_loss / len(params.train_loader)
	)
    return

def validate(model,
             params: train_params):
    model.eval()
    total_ade, total_fde, total_loss = 0, 0, 0
    count = 0

    with torch.no_grad():
        for batch in params.val_loader:
            cam, hist, fut = [batch[k].to(params.device) for k in ['camera', 'history', 'future']]

            fut_pred = model(cam, hist)

            B, _, _ = fut.shape
            count += B

            loss = params.criterion(fut_pred, fut)
            ade = torch.norm(fut_pred[:, :, :2] - fut[:, :, :2], dim=2).mean(dim=1).sum()
            fde = torch.norm(fut_pred[:, -1, :2] - fut[:, -1, :2], dim=1).sum()

            total_ade += ade.item()
            total_fde += fde.item()
            total_loss += loss.item()

    ade_avg = total_ade / count
    fde_avg = total_fde / count
    mse_avg = total_loss / len(params.val_loader)

    params.logger.log(ADE=ade_avg, FDE=fde_avg, loss_val=mse_avg)

    if (params.save_best) and (params.logger.best_ade == ade_avg):
        torch.save(model.state_dict(), params.save_best_fname)
        print(f"Saving model at epoch {params.current_epoch+1} with ADE: {ade_avg}")

def train(model,
          params: train_params):
    print(f"Using device: {params.device}")
    model = model.to(params.device)

    best_ADE = 50

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    os.makedirs("results", exist_ok=True)
    filename = f"results/output_{timestamp}.pt"
    params.set_save_best(filename)
    print("Save name is: ", filename)

    for epoch in range(params.tot_epochs):
        train_one_epoch(model, params)
        validate(model, params)
        
        params.logger.plot()
        params.logger.printf()
        if params.scheduler is not None:
            params.scheduler.step()
        params.increase_epoch()