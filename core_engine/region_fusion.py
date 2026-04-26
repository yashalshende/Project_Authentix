import torch
import torch.nn as nn

try:
    from core_engine.config import ModelConfig as cfg
except ImportError:
    from config import ModelConfig as cfg


class FaceRegionFeatureEncoder(nn.Module):
    """
    Shared lightweight encoder for 3x3 facial regions.
    Keeps the branch practical for localhost training while exposing region scores.
    """

    def __init__(self, in_channels=3, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or cfg.REGION_EMBED_DIM

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, embed_dim)
        self.region_score_head = nn.Linear(embed_dim, 1)
        self.region_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, region_crops):
        """
        region_crops shape: (B, 9, C, H, W)
        returns:
          fused_embedding: (B, embed_dim)
          region_scores: (B, 9)
          attention_weights: (B, 9)
        """
        batch_size, num_regions, channels, height, width = region_crops.shape
        flat_regions = region_crops.view(batch_size * num_regions, channels, height, width)
        features = self.backbone(flat_regions).view(batch_size * num_regions, -1)
        embeddings = self.projection(features)

        region_scores = self.region_score_head(embeddings).view(batch_size, num_regions)
        attention_logits = self.region_attention(embeddings).view(batch_size, num_regions)
        attention_weights = torch.softmax(attention_logits, dim=1)

        embeddings = embeddings.view(batch_size, num_regions, -1)
        fused_embedding = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)
        return fused_embedding, region_scores, attention_weights
