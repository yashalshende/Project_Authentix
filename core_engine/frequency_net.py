import torch
import torch.nn as nn
import torch.nn.functional as F

#DWT
class HaarDWT2D(nn.Module):
    """
    Calculates Discrete Wavelet Transforms natively inside the GPU by executing
    Stride=2 manual Convolutions utilizing strictly constant Haar filters.
    Avoids slow redundant Numpy CPU -> Torch GPU tensor offloads.
    """
    def __init__(self):
        super(HaarDWT2D, self).__init__()
        
        # Standardized Haar mathematical frequency constants
        ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
        lh = torch.tensor([[-1., -1.], [1., 1.]]) / 2.0
        hl = torch.tensor([[-1., 1.], [-1., 1.]]) / 2.0
        hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
        
        filter_bank = torch.stack([ll, lh, hl, hh]).unsqueeze(1) # Final mapping: (4, 1, 2, 2)
        
        # Register the mathematically defined static arrays avoiding autograd gradients 
        self.register_buffer('filter_bank', filter_bank)

    def forward(self, x):
        # Maps (Batch, Channel, Height, Width)
        B, C, H, W = x.shape
        
        # Shatter RGB channels into independent singular evaluations dynamically
        x = x.view(B * C, 1, H, W)
        
        # Stride=2 performs native downsampling during the pass
        out = F.conv2d(x, self.filter_bank, stride=2, padding=0)
        
        # Consolidate back to mapping configurations extracting 4 representations per channel
        # 3 Channels (RGB) * 4 Matrices (LL, LH, HL, HH) = 12 Channels Total Output
        out = out.view(B, C * 4, H // 2, W // 2)
        return out

#CNN on top of DWT
class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=3, out_features=256):
        """
        Siphons underlying block-noise compression irregularities left by generic upsampling mechanisms
        used during facial decoding (GAN decoders).
        """
        super(FrequencyBranch, self).__init__()
        
        self.dwt = HaarDWT2D()
        
        # Output sub-bands evaluates strictly through an independent lightweight CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels * 4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            nn.Linear(128, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        freq_features = self.dwt(x)
        out = self.cnn(freq_features)
        return out
