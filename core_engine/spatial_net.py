import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

#CNN
class SpatialBranch(nn.Module):
    def __init__(self, pretrained=True):
        """
        Calculates pixel-intensity structural errors globally spanning the physical RGB spectrum.
        """
        super(SpatialBranch, self).__init__()
        
        # Lock internal pre-defined ImageNet mapping weights optimally
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        effnet = efficientnet_v2_s(weights=weights)
        
        # Halt execution directly prior to classification pooling.
        # This preserves the (B, 1280, H', W') convolutional tensor inherently allowing Grad-CAM
        # to mathematically attach to these spatial feature distributions.
        self.features = effnet.features
        
        # Feature geometric mapping depth for V2-S = 1280
        self.out_channels = 1280

    def forward(self, x):
        return self.features(x)
