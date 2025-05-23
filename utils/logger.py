import matplotlib.pyplot as plt
import sys
import numpy as np

class Logger:
    def __init__(self, tot_epochs):
        self.loss_val = []
        self.loss_train  = []
        self.ade_val  = []
        self.fde_val  = []
        self.best_ade = 50

        self.epoch    = []
        self.total_epochs = tot_epochs

    def log(self, epoch=None, **metrics):
        if epoch is not None:
            self.epoch.append(epoch + 1)

        for key, val in metrics.items():
            if key == 'loss_val':
                self.loss_val.append(val)
            elif key == 'loss_train':
                self.loss_train.append(val)
            elif key == 'ADE':
                self.ade_val.append(val)
                if val < self.best_ade:
                    self.best_ade = val
            elif key == 'FDE':
                self.fde_val.append(val)

    def plot(self):
        if self.epoch[-1] == 1:
            return
        fig, axs = plt.subplots(1, 5, figsize=(25, 5))

        axs[0].plot(self.epoch, self.loss_train, marker='o', color='blue')
        axs[0].set_title("Training Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss [-]")
        axs[0].grid(True)

        axs[1].plot(self.epoch, self.loss_val, marker='x', color='red')
        axs[1].set_title("Validation Loss")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Loss [-]")
        axs[1].grid(True)

        axs[2].plot(self.epoch, self.ade_val, marker='x', color='red')
        axs[2].set_title("Validation ADE")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Error [-]")
        axs[2].grid(True)

        axs[3].plot(self.epoch, self.fde_val, marker='o', color='blue')
        axs[3].set_title("Validation FDE")
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("Error [-]")
        axs[3].grid(True)

        axs[4].plot(self.epoch, np.array(self.loss_val) - np.array(self.loss_train), marker='o', color='green')
        axs[4].set_title("Loss Difference")
        axs[4].set_xlabel("Epoch")
        axs[4].set_ylabel("Val - Train Loss [-]")
        axs[4].grid(True)

        fig.tight_layout()
        fig.savefig("Train_Progress.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def printf(self):
        message = f'Epoch {self.epoch[-1]}/{self.total_epochs}'
        message += f' | Train Loss: {self.loss_train[-1]:.4f}'
        message += f' | Val Loss: {self.loss_val[-1]:.4f}'
        message += f' | ADE: {self.ade_val[-1]:.4f}'
        message += f' | FDE: {self.fde_val[-1]:.4f}'
        message += f' | Best ADE: {self.best_ade:.4f}'

        sys.stdout.write(f'\n{message}\n')
        sys.stdout.flush()

        if self.epoch[0] == 1:
            self.loss_val.pop(0)
            self.loss_train.pop(0)
            self.ade_val.pop(0)
            self.fde_val.pop(0)
            self.epoch.pop(0)