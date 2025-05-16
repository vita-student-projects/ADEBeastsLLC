import matplotlib.pyplot as plt
import random
import utils.dataset as d
import torch
import numpy as np

from utils.train import compute_losses

def visualize_with_depth(val_loader, model, device):
    model.eval()
    val_batch = next(iter(val_loader))

    camera = val_batch['camera'].to(device)
    history = val_batch['history'].to(device)
    future = val_batch['future'].to(device)
    depth = val_batch['depth'].to(device)
    sem_gt = val_batch['semantic_label'].to(device)

    with torch.no_grad():
        pred_future, pred_depth, pred_sem = model(camera, history)

    camera = camera.cpu().numpy()
    history = history.cpu().numpy()
    future = future.cpu().numpy()
    pred_future = pred_future.cpu().numpy()
    depth = depth.cpu().numpy()
    pred_depth = pred_depth.cpu().numpy() if pred_depth is not None else None
    semantic_gt = sem_gt.cpu().numpy()
    pred_semantic = pred_sem.cpu().numpy() if pred_sem is not None else None

    k = 4
    indices = random.choices(np.arange(len(camera)), k=k)

    # Show the input camera images
    fig, ax = plt.subplots(6, k, figsize=(4 * k, 3*6))
    for i, idx in enumerate(indices):
        ax[0, i].imshow(d.unnormalize_img(camera[idx], d.img_mean, d.img_std))
        ax[0, i].set_title(f"Example {i+1}")
        ax[0, i].axis("off")
    
        # Trajectory prediction
        ax[1, i].plot(history[idx, :, 0], history[idx, :, 1], 'o-', label='Past', color='gold', markersize=4, linewidth=1.2)
        ax[1, i].plot(future[idx, :, 0], future[idx, :, 1], 'o-', label='GT Future', color='green', markersize=4, linewidth=1.2)
        ax[1, i].plot(pred_future[idx, :, 0], pred_future[idx, :, 1], 'o-', label='Pred (Future)', color='red', markersize=4, linewidth=1.2)
        ax[1, i].axis("equal")

        # Depth True
        ax[2, i].imshow(depth[idx, 0, :, :], cmap='viridis')
        ax[2, i].set_title("GT Depth", pad=10)
        ax[2, i].axis("off")

        # Depth Pred
        if pred_depth is not None:
            ax[3, i].imshow(pred_depth[idx, 0, :, :], cmap='viridis')
        ax[3, i].set_title("Pred Depth", pad=10)
        ax[3, i].axis("off")

        # Plot semantic image
        plot_semantic = semantic_gt[idx]
        plot_semantic = d.semantic_class_to_color(plot_semantic[0])
        ax[4, i].imshow(plot_semantic.permute(1,2,0))
        ax[4, i].set_title(f"GT Segmentation")
        ax[4, i].axis("off")

        # semantic predicted
        if pred_semantic is not None:
            plot_semantic = pred_semantic[idx]
            plot_semantic = np.argmax(plot_semantic, axis=0)
            plot_semantic = d.semantic_class_to_color(plot_semantic)
            ax[5, i].imshow(plot_semantic.permute(1,2,0))
        ax[5, i].set_title(f"Pred Segmentation")
        ax[5, i].axis("off")

    plt.tight_layout()
    plt.savefig("Model_Visual_Eval.png", dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model,
                   val_loader,
                   device):
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

    message = "\nModel Validation Performance:\n"
    message += f"\tADE: {ade_avg:.4f}\n"
    message += f"\tFDE: {fde_avg:.4f}\n"
    message += f"\tLoss: {mse_avg:.4f}\n"
    print(message)

    return