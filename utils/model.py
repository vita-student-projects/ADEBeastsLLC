import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class DrivingPlanner_GRU(nn.Module):
    def __init__(self): # Changed _init to _init_
        super().__init__()

        # Load pretrained ResNet
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Freeze all resnet parameters
        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # for param in resnet.layer3.parameters():
        #     param.requires_grad = True

        # Use all resnet layers up to (but not including) avgpool
        self.img_encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 56 x 56 x 64
            resnet.layer2, # 28 x 28 x 128
            resnet.layer3, # 14 x 14 x 256
            resnet.layer4, # Output: 7x7x512
            resnet.avgpool,
            nn.Flatten(),  # Output: batchx512
            nn.Linear(512, 256)
        )

        # Encoder for the history
        self.history_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*21, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Decoder for predicting future trajectory
        self.future_gru     = nn.GRU(
            input_size=3,
            hidden_size=(128+256),
            batch_first=True
        )
        self.future_decoder = nn.Sequential(
            nn.Linear((128+256), 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, camera, history):
        # Process camera images
        img_enc = self.img_encoder(camera)

        # Process History
        history_encoded = self.history_enc(history)

        # hiddest state
        h0 = torch.cat([img_enc, history_encoded], dim=1).unsqueeze(0) #(1, batch, hidden_size)

        # Predict future
        future_traj = []
        curr_pos = history[:, -1, :].unsqueeze(1) # (batch, 1, 3)
        for idx in range(60):
            # GRU forward
            out, h0 = self.future_gru(curr_pos, h0) # out (batch, 1, hidden_size)

            # Predict next value
            next_pos = self.future_decoder(out) # (batch, 1, 3)
            future_traj.append(next_pos)

            # Update current pos
            curr_pos = next_pos
        future_traj = torch.cat(future_traj, dim=1) # (batch, 60, 3)
        return future_traj

class DrivingPlanner(nn.Module):
    def __init__(self): # Changed _init to _init_
        super().__init__()

        # Load pretrained ResNet
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Freeze all resnet parameters
        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # for param in resnet.layer3.parameters():
        #     param.requires_grad = True

        # Use all resnet layers up to (but not including) avgpool
        self.img_encoder_L1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 56 x 56 x 64
        )
        self.img_encoder_L2 = resnet.layer2 # 28 x 28 x 128
        self.img_encoder_L3 = resnet.layer3 # 14 x 14 x 256
        self.img_encoder_L4 = resnet.layer4 # Output: 7x7x512


        # Use Resnet
        self.resnet = nn.Sequential(
            resnet.avgpool,
            nn.Flatten(),  # Output: batchx512
            nn.Linear(512, 256)
        )

        # Encoder for the history
        self.history_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*21, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Decoder for predicting future trajectory
        self.future_decoder_1  = nn.GRU(input_size=(128+256), hidden_size=128, batch_first=True)
        self.future_decoder_2  = nn.GRU(input_size=128, hidden_size=64, batch_first=True)
        self.output_trajectory = nn.Linear(64, 3)

        self.future_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 60 * 3)
        )

    def forward(self, camera, history):
        # Process camera images
        img_encoder_L1 = self.img_encoder_L1(camera)
        img_encoder_L2 = self.img_encoder_L2(img_encoder_L1)
        img_encoder_L3 = self.img_encoder_L3(img_encoder_L2)
        visual_features = self.img_encoder_L4(img_encoder_L3)
        visual_resnet   = self.resnet(visual_features)

        # Process History
        history_encoded = self.history_enc(history)

        # Predict future
        combined = torch.cat([visual_resnet, history_encoded], dim=1)
        future = self.future_decoder(combined)
        future = future.reshape(-1, 60, 3)  # Reshape to (batch_size, timesteps, features)

        # Voila
        return future