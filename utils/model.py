import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class DrivingPlanner(nn.Module):
    def __init__(self, use_depth_aux=False, use_semantic_aux=False): # Changed _init to _init_
        super().__init__()
        self.use_depth_aux = use_depth_aux
        self.use_semantic_aux = use_semantic_aux

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

        # Decoder for Depth estimation
        if self.use_depth_aux:
            self.depth_L3 = self._upsample_block(512, 256)       # 14 x 14   x 256
            self.depth_L2 = self._upsample_block(256 + 256, 128) # 28 x 28   x 128
            self.depth_L1 = self._upsample_block(128 + 128, 64)  # 56 x 56   x 64
            self.depth_decoder = nn.Sequential(
                self._upsample_block(64 + 64, 32),    # 112 x 112 x 32
                self._upsample_block(32, 16),         # 224 x 224 x 16
                nn.Conv2d(16, 1, kernel_size=3, padding=1),  # 224 x 224 x 1
            )

        if self.use_semantic_aux:
            self.semantic_L3 = self._upsample_block(512, 256)       # 14 x 14   x 256
            self.semantic_L2 = self._upsample_block(256 + 256, 128) # 28 x 28   x 128
            self.semantic_L1 = self._upsample_block(128 + 128, 64)  # 56 x 56   x 64
            self.semantic_decoder = nn.Sequential(
                self._upsample_block(64 + 64, 32),    # 112 x 112 x 32
                self._upsample_block(32, 16),         # 224 x 224 x 16
                nn.Conv2d(16, 15, kernel_size=3, padding=1),  # 224 x 224 x 3
            )

        # Encoder for the history
        self.history_encoder_1 = nn.GRU(input_size=3, hidden_size=256, batch_first=True, bidirectional=True)
        self.history_encoder_2 = nn.GRU(input_size=256, hidden_size=128, batch_first=True)
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

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

        # Predict depth
        if self.use_depth_aux:
            depth_L3 = self.depth_L3(visual_features)
            depth_L2 = self.depth_L2(torch.cat([depth_L3, img_encoder_L3], dim=1))
            depth_L1 = self.depth_L1(torch.cat([depth_L2, img_encoder_L2], dim=1))
            depth_out = self.depth_decoder(torch.cat([depth_L1, img_encoder_L1], dim=1))

        # Predict semantic information
        if self.use_semantic_aux:
            semantic_L3 = self.semantic_L3(visual_features)
            semantic_L2 = self.semantic_L2(torch.cat([semantic_L3, img_encoder_L3], dim=1))
            semantic_L1 = self.semantic_L1(torch.cat([semantic_L2, img_encoder_L2], dim=1))
            semantic_out = self.semantic_decoder(torch.cat([semantic_L1, img_encoder_L1], dim=1))

        # Voila
        if self.use_depth_aux and self.use_semantic_aux:
            return future, depth_out, semantic_out
        elif self.use_depth_aux:
            return future, depth_out, None
        elif self.use_semantic_aux:
            return future, None, semantic_out
        else:
            return future, None, None



class FirstModel(nn.Module):
    def __init__(self): # Changed _init_ to __init__
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),    # C x (100x150)
            nn.ReLU(),
            nn.Conv2d(16, 128, kernel_size=5, stride=2, padding=2),  # C x (50x75)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),        # C x (25x38)
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # C x (13x19)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2), # C x (7x10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),        # C x (4x5)
            nn.Flatten(),
            nn.Linear(3 * 5 * 512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )

        self.history = nn.Sequential(
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
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 60 * 3),
        )

    def forward(self, camera, history):
        # Process camera images
        visual_features = self.cnn(camera)

        # Combine features
        history_latent = self.history(history)

        combined = torch.cat([visual_features, history_latent], dim=1)

        # Predict future trajectory
        future = self.decoder(combined)

        future = future.reshape(-1, 60, 3)  # Reshape to (batch_size, timesteps, features)

        return future