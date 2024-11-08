# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class CustomCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(CustomCNN, self).__init__()
        # CNN Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        # Calculate feature dimensions
        self.feature_dims = self._get_conv_output_dims()
        
        # ResNet feature extractor
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
            
        # Combine features
        self.combine_features = nn.Sequential(
            nn.Linear(self.feature_dims + 1000, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def _get_conv_output_dims(self):
        x = torch.randn(1, 3, 64, 64)
        x = self.conv_layers(x)
        return x.numel() // x.size(0)

    def forward(self, x):
        cnn_features = self.conv_layers(x)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        resnet_features = self.resnet(x)
        combined = torch.cat((cnn_features, resnet_features), dim=1)
        return self.combine_features(combined)