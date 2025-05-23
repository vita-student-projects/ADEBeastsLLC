import matplotlib.pyplot as plt
import random
from utils.dataset import unnormalize_img
import torch
import numpy as np

def visualize(val_loader, model, device):
    model.eval()

    val_batch_zero = next(iter(val_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera = val_batch_zero['camera'].to(device)
    history = val_batch_zero['history'].to(device)
    future = val_batch_zero['future'].to(device)

    with torch.no_grad():
        pred_future = model(camera, history)

    camera = camera.cpu().numpy()
    history = history.cpu().numpy()
    future = future.cpu().numpy()
    pred_future = pred_future.cpu().numpy()

    k = 4
    indices = random.choices(np.arange(len(camera)), k=k)

    # Show the input camera images
    fig, ax = plt.subplots(2, k, figsize=(4 * k, 3*2))
    for i, idx in enumerate(indices):
        ax[0, i].imshow(unnormalize_img(camera[idx]))
        ax[0, i].set_title(f"Example {i+1}")
        ax[0, i].axis("off")
    
        # Trajectory prediction
        ax[1, i].plot(history[idx, :, 0], history[idx, :, 1], 'o-', label='Past', color='gold', markersize=4, linewidth=1.2)
        ax[1, i].plot(future[idx, :, 0], future[idx, :, 1], 'o-', label='GT Future', color='green', markersize=4, linewidth=1.2)
        ax[1, i].plot(pred_future[idx, :, 0], pred_future[idx, :, 1], 'o-', label='Pred (Future)', color='red', markersize=4, linewidth=1.2)
        ax[1, i].axis("equal")
        ax[1, i].legend()

    plt.tight_layout()
    plt.savefig("Model_Visual_Eval.png", dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model,
                   params):
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

    message = "\nModel Validation Performance:\n"
    message += f"\tADE: {ade_avg:.4f}\n"
    message += f"\tFDE: {fde_avg:.4f}\n"
    message += f"\tLoss: {mse_avg:.4f}\n"
    print(message)

    return