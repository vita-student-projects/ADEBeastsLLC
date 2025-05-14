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
  

traj_criterion = customCriterion()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
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

        traj_loss = traj_criterion(fut_pred, fut)
        loss = traj_loss
        total_trajectory_loss += traj_loss.item()
        if use_depth_aux:
            depth_loss = lambda_depth * F.l1_loss(dep_pred, dep)
            loss += depth_loss
            depth_loss += depth_loss
        if use_semantic_aux:
            semantic_loss = lambda_semantic * F.l1_loss(sem_pred, sem)
            loss += semantic_loss

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if use_depth_aux:
          total_depth_loss += depth_loss.item()
          avg_depth_loss = total_depth_loss / len(train_loader)
        else:
          avg_depth_loss = 0

        if use_semantic_aux:
          total_semantic_loss += semantic_loss.item()
          avg_semantic_loss = total_semantic_loss / len(train_loader)
        else:
          avg_semantic_loss = 0

    avg_trajectory_loss = total_trajectory_loss / len(train_loader)
    avg_loss = train_loss / len(train_loader)
    logger.log(epoch=epoch, total_loss=avg_loss, traj_loss=avg_trajectory_loss, depth_loss = avg_depth_loss, semantic_loss = avg_semantic_loss)
    return

def validate(model,
             val_loader,
             device,
             logger,
             best_ADE,
             model_save_path,
             epoch=None):
    model.eval()
    total_ade, total_fde, total_mse = 0.0, 0.0, 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            cam = batch['camera'].to(device)
            hist = batch['history'].to(device)
            fut = batch['future'].to(device)

            fut_pred, _, _ = model(cam, hist)

            B, T, _ = fut.shape
            count += B

            ade = torch.norm(fut_pred[:, :, :2] - fut[:, :, :2], dim=2).mean(dim=1).sum()
            fde = torch.norm(fut_pred[:, -1, :2] - fut[:, -1, :2], dim=1).sum()
            mse = F.mse_loss(fut_pred, fut, reduction='sum')

            total_ade += ade.item()
            total_fde += fde.item()
            total_mse += mse.item()

    ade_avg = total_ade / count
    fde_avg = total_fde / count
    mse_avg = total_mse / (count * T * 3)

    if epoch is not None:
      logger.log(ADE=ade_avg, FDE=fde_avg, loss_val=mse_avg)

    if ade_avg < best_ADE:
       torch.save(model.save_dict(), model_save_path)
       print(f"Saving model at epoch {epoch} with ADE: {ade_avg}")

    return best_ADE, mse

def train(model,
          train_loader,
          val_loader,
          optimizer,
          logger,
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

    best_ADE = 1.8

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    filename = f"output_{timestamp}"

    earlystopping = EarlyStopping(patience=10, delta=0.01)

    for epoch in range(num_epochs):
        model_save_path = filename + "_epoch_" + epoch + ".pt"
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
        best_ADE, mse = validate(model, val_loader, device, logger, best_ADE, model_save_path, epoch)
        if earlystopping(mse, model):
           print("early stopping")
           break
        
        # logger.plot()
        logger.printf()
        if epoch == 0 and start_epoch == 0:
            logger.clean()
        if scheduler is not None:
            scheduler.step()