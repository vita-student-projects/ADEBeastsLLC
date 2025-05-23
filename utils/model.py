import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

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