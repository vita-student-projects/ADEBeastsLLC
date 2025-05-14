import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

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
  
## Defining the criterions
traj_criterion = customCriterion()
semantic_criterion = nn.L1Loss() # Use cross entropy if using semantic label not image
depth_criterion = nn.L1Loss()

def compute_losses(traj, dep, sem,
                   lambda_sem=0.1, lambda_depth=0.1,
                   use_depth_aux=False, use_semantic_aux=False):
	# Calculate losses
	custom_loss   = traj_criterion(traj[0], traj[1])
	depth_loss    = lambda_depth*depth_criterion(dep[0], dep[1]) if use_depth_aux else 0

	true_sem = sem[1]
  # print(sem[0].shape)
	semantic_loss = lambda_sem*semantic_criterion(sem[0], true_sem)  if use_semantic_aux else 0

	# Track losses
	return [custom_loss, depth_loss, semantic_loss]

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

########################## TRAINING FUNCTIONS #######################

def train_one_epoch(model,
                    train_loader,
                    optimizer,
                    device,
                    logger,
                    epoch,
                    lambda_depth=0.1,
                    lambda_semantic=0.1,
                    use_depth_aux=False,
                    use_semantic_aux=False):
    model.train()

    train_loss = 0.0
    total_semantic_loss = 0.0
    total_depth_loss = 0.0
    total_trajectory_loss = 0.0

    for batch in train_loader:
        cam, hist, fut, dep, sem = [batch[k].to(device) for k in ['camera', 'history', 'future', 'depth', 'semantic_label']]
        optimizer.zero_grad()
        fut_pred, dep_pred, sem_pred = model(cam, hist)

        losses = compute_losses([fut_pred, fut],
                                [dep_pred, dep],
                                [sem_pred, sem],
                                use_depth_aux=use_depth_aux,
                                use_semantic_aux=use_semantic_aux,
                                lambda_sem=lambda_semantic,
                                lambda_depth=lambda_depth)

        traj_loss = traj_criterion(fut_pred, fut)
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_trajectory_loss += losses[0].item()
        total_depth_loss += losses[1]
        total_semantic_loss += losses[2]

        logger.log(epoch=epoch,
          total_loss=train_loss / len(train_loader),
          traj_loss=total_trajectory_loss / len(train_loader),
          depth_loss=total_depth_loss / len(train_loader),
          semantic_loss=total_semantic_loss / len(train_loader)
        )
    return

def validate(model,
             val_loader,
             device,
             logger,
             best_ADE,
             model_save_path,
             epoch=None,
             post_train=False):
    model.eval()
    total_ade, total_fde, total_mse = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            cam, hist, fut, dep, sem = [batch[k].to(device) for k in ['camera', 'history', 'future', 'depth', 'semantic_label']]

            fut_pred, dep_pred, sem_pred = model(cam, hist)

            B, T, _ = fut.shape
            count += B

            losses = compute_losses([fut_pred, fut],
                                    [dep_pred, dep],
                                    [sem_pred, sem],
                                    lambda_sem=0,
                                    lambda_depth=0)
            ade = torch.norm(fut_pred[:, :, :2] - fut[:, :, :2], dim=2).mean(dim=1).sum()
            fde = torch.norm(fut_pred[:, -1, :2] - fut[:, -1, :2], dim=1).sum()

            total_ade += ade.item()
            total_fde += fde.item()
            total_mse += losses[0].item()

    ade_avg = total_ade / count
    fde_avg = total_fde / count
    mse_avg = total_mse / len(val_loader)

    if epoch is not None:
      logger.log(ADE=ade_avg, FDE=fde_avg, loss_val=mse_avg)

    if ade_avg < best_ADE and not post_train:
      best_ADE = ade_avg
      torch.save(model.state_dict(), model_save_path)
      print(f"Saving model at epoch {epoch+1} with ADE: {ade_avg}")

    return best_ADE, mse_avg, ade_avg

def train(model,
          train_loader,
          val_loader,
          optimizer,
          logger,
          model_save_path,
          num_epochs=50,
          start_epoch=0,
          lambda_depth=0.1,
          lambda_semantic=0.1,
          use_depth_aux=False,
          use_semantic_aux=False,
          scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    best_ADE = 1.70

    early_stopping = EarlyStopping(patience=7, delta=0.005)

    for epoch in range(num_epochs):
        train_one_epoch(model,
                        train_loader,
                        optimizer,
                        device,
                        logger,
                        epoch,
                        lambda_depth,
                        lambda_semantic,
                        use_depth_aux,
                        use_semantic_aux)
      
        best_ADE, val_loss, _ = validate(model, val_loader, device, logger, best_ADE, model_save_path, epoch)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
           print("early stopping")
           early_stopping.load_best_model(model)
           break
        # logger.plot()
        logger.printf()
        if epoch == 0 and start_epoch == 0:
            logger.clean()
        if scheduler is not None:
            scheduler.step()


