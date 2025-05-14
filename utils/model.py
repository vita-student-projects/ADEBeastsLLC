import torch
import torch.nn as nn
from torchvision import models


class DrivingPlanner(nn.Module):
    def __init__(self, use_depth_aux=False, use_semantic_aux=False): # Changed _init to _init_
        super().__init__()
        self.use_depth_aux = use_depth_aux
        self.use_semantic_aux = use_semantic_aux

        # Load pretrained ResNet
        resnet = models.resnet34(pretrained=True)

        # Freeze all resnet parameters
        for param in resnet.parameters():
            param.requires_grad = False

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
        self.history_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*21, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Decoder for predicting future trajectory
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

        # Decoder for Depth estimation
        if self.use_depth_aux:
            self.depth_L3 = self._upsample_block(512, 256)       # 14 x 14   x 256
            # self.depth_L3_lin
            self.depth_L2 = self._upsample_block(256 + 256, 128) # 28 x 28   x 128
            self.depth_L1 = self._upsample_block(128 + 128, 64)  # 56 x 56   x 64
            self.depth_decoder = nn.Sequential(
                self._upsample_block(64 + 64, 32),    # 112 x 112 x 32
                self._upsample_block(32, 8),         # 224 x 224 x 16
                # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224 -> 448x448
                # nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),  # 448 x 448 x 1
            )

        if self.use_semantic_aux:
            self.semantic_L3 = self._upsample_block(512, 256)       # 14 x 14   x 256
            self.semantic_L2 = self._upsample_block(256 + 256, 128) # 28 x 28   x 128
            self.semantic_L1 = self._upsample_block(128 + 128, 64)  # 56 x 56   x 64
            self.semantic_decoder = nn.Sequential(
                self._upsample_block(64 + 64, 32),    # 112 x 112 x 32
                self._upsample_block(32, 8),         # 224 x 224 x 16
                # nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224 -> 448x448
                # nn.ReLU(inplace=True),
                nn.Conv2d(8, 3, kernel_size=3, padding=1),  # 448 x 448 x 3
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
        history = self.history_encoder(history)

        # Predict future
        combined = torch.cat([visual_resnet, history], dim=1)
        future = self.future_decoder(combined)
        future = future.reshape(-1, 60, 3)  # Reshape to (batch_size, timesteps, features)

        # Predict depth
        if self.use_depth_aux:
            depth_L3 = self.depth_L3(visual_features)
            depth_L2 = self.depth_L2(torch.cat([depth_L3, img_encoder_L3], dim=1))
            depth_L1 = self.depth_L1(torch.cat([depth_L2, img_encoder_L2], dim=1))
            depth_out = self.depth_decoder(torch.cat([depth_L1, img_encoder_L1], dim=1))
            # depth_out = F.interpolate(depth_out, size=(224, 224), mode='area').squeeze(1)#.permute(0, 2, 3, 1)
            # depth_out = depth_out[:,:200, :300, :]

        # Predict semantic information
        if self.use_semantic_aux:
            semantic_L3 = self.semantic_L3(visual_features)
            semantic_L2 = self.semantic_L2(torch.cat([semantic_L3, img_encoder_L3], dim=1))
            semantic_L1 = self.semantic_L1(torch.cat([semantic_L2, img_encoder_L2], dim=1))
            semantic_out = self.semantic_decoder(torch.cat([semantic_L1, img_encoder_L1], dim=1))
            # semantic_out = F.interpolate(semantic_out, size=(224, 224), mode='area')#.permute(0, 2, 3, 1)
            # semantic_out = semantic_out[:,:,:200, :300]

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