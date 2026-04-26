import torch
import torch.nn as nn

try:
    from core_engine.config import ModelConfig as cfg
    from core_engine.spatial_net import SpatialBranch
    from core_engine.frequency_net import FrequencyBranch
    from core_engine.attention import CBAM
    from core_engine.region_fusion import FaceRegionFeatureEncoder
except ImportError:
    # Handling logical fallback instances when evaluated natively from root CLI
    from config import ModelConfig as cfg
    from spatial_net import SpatialBranch
    from frequency_net import FrequencyBranch
    from attention import CBAM
    from region_fusion import FaceRegionFeatureEncoder

class AuthentixHybridModel(nn.Module):
    def __init__(self):
        """
        Unified deep classification engine seamlessly linking pixel-geometric structures
        organically to high-frequency GAN noise arrays natively under parallel convergence.
        """
        super(AuthentixHybridModel, self).__init__()
        
        # 1. Primary RGB Evaluation Branch
        self.spatial_branch = SpatialBranch(pretrained=True)
        
        # 2. CBAM: Convolutional Block Attention Module
        self.use_cbam = cfg.USE_CBAM
        if self.use_cbam:
            self.spatial_attention = CBAM(self.spatial_branch.out_channels)
            
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_flatten = nn.Flatten()
        
        # 3. Micro-Subband DWT Extraction Branch
        self.freq_branch = FrequencyBranch(in_channels=cfg.IN_CHANNELS, out_features=cfg.FREQ_FEATURES_DIM)
        self.use_face_region_branch = getattr(cfg, "USE_FACE_REGION_BRANCH", True)
        self.region_branch = FaceRegionFeatureEncoder() if self.use_face_region_branch else None
        self.use_face_swap_aux_features = getattr(cfg, "USE_FACE_SWAP_AUX_FEATURES", True)
        self.aux_projection = (
            nn.Sequential(
                nn.Linear(cfg.FACE_SWAP_AUX_DIM, 64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
            )
            if self.use_face_swap_aux_features
            else None
        )

        # 4. Neural Tensor Fusion Head
        # Output dimensional maps: EfficientNetB0 (1280) + Frequency Length (256) = 1536 Matrix Length
        region_dim = cfg.REGION_EMBED_DIM if self.use_face_region_branch else 0
        aux_dim = 64 if self.use_face_swap_aux_features else 0
        fusion_input_dim = self.spatial_branch.out_channels + cfg.FREQ_FEATURES_DIM + region_dim + aux_dim
        
        self.fusion_embedder = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.FUSION_DIM),
            nn.BatchNorm1d(cfg.FUSION_DIM),
            nn.ReLU(inplace=True)
        )
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(cfg.FUSION_DIM, cfg.NUM_CLASSES)
        )
        self.faceswap_head = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE * 0.8),
            nn.Linear(cfg.FUSION_DIM, 1)
        )
        self.return_embedding = False  # Toggle for Temporal LSTM video execution

    def forward(self, x, region_crops=None, aux_features=None):
        # Process Spatial Features natively -> Returning shape (Batch, 1280, H', W')
        spatial_feat_maps = self.spatial_branch(x)
        
        # Highlight boundary faults mathematically leveraging attention networks
        if self.use_cbam:
            spatial_feat_maps = self.spatial_attention(spatial_feat_maps)
            
        # Global Avg Pooling execution resolving map to linear -> Shape (Batch, 1280)
        spatial_features = self.spatial_flatten(self.spatial_pool(spatial_feat_maps))
        
        # Evaluate localized wavelets exclusively resolving map to -> Shape (Batch, 256)
        freq_features = self.freq_branch(x)
        
        # Vector Combination -> Yielding Length (Batch, 1536)
        region_scores = None
        attention_weights = None
        if self.use_face_region_branch and region_crops is not None:
            region_embedding, region_scores, attention_weights = self.region_branch(region_crops)
            fused = torch.cat((spatial_features, freq_features, region_embedding), dim=1)
        else:
            if self.use_face_region_branch:
                zeros = torch.zeros(
                    spatial_features.size(0),
                    cfg.REGION_EMBED_DIM,
                    device=spatial_features.device,
                    dtype=spatial_features.dtype,
                )
                fused = torch.cat((spatial_features, freq_features, zeros), dim=1)
            else:
                fused = torch.cat((spatial_features, freq_features), dim=1)

        if self.use_face_swap_aux_features:
            if aux_features is None:
                aux_features = torch.zeros(
                    spatial_features.size(0),
                    cfg.FACE_SWAP_AUX_DIM,
                    device=spatial_features.device,
                    dtype=spatial_features.dtype,
                )
            aux_embedding = self.aux_projection(aux_features)
            fused = torch.cat((fused, aux_embedding), dim=1)

        # Dense mathematical dimensional compression -> Yielding (Batch, 512)
        embedding = self.fusion_embedder(fused)
        
        # Binary Classification outcome -> Unbounded Logit Tensor
        # Using BCEWithLogitsLoss seamlessly applies sigmoid mathematical translation during backpropagation
        output = self.fusion_classifier(embedding)
        faceswap_logits = self.faceswap_head(embedding)

        # Yield internal embedding natively bypassing standard structural classifiers 
        if getattr(self, 'return_embedding', False):
            return output, spatial_feat_maps, embedding, region_scores, attention_weights, faceswap_logits

        # Return both the outcome score AND the core spatial mapping array unconditionally
        return output, spatial_feat_maps, region_scores, attention_weights, faceswap_logits

# Quick Local Integrity Check ensuring dimension matrices align perfectly.
if __name__ == "__main__":
    print("Initiating AUTHENTIX Network verification...")
    # Generate completely dummy data modeling precisely a standard single image (1, RGB, 256, 256)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Initialize Core Network
    model = AuthentixHybridModel()
    
    # Run Inference
    dummy_regions = torch.randn(1, cfg.NUM_FACE_REGIONS, 3, cfg.REGION_SIZE, cfg.REGION_SIZE)
    dummy_aux = torch.randn(1, cfg.FACE_SWAP_AUX_DIM)
    logits, cam_feature_map, region_scores, attention_weights, faceswap_logits = model(dummy_input, dummy_regions, dummy_aux)
    
    print(f"Total Parameter Density: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million")
    print(f"Logits Matrix Output Shape: {logits.shape} (Expected: 1, 1)")
    print(f"Grad-CAM Extraction Matrix Shape: {cam_feature_map.shape} (Expected: 1, 1280, 8, 8)")
    print(f"Region Score Matrix Shape: {region_scores.shape if region_scores is not None else None} (Expected: 1, 9)")
    print(f"Face Swap Head Output Shape: {faceswap_logits.shape if faceswap_logits is not None else None} (Expected: 1, 1)")
    print("AUTHENTIX successfully compiled dimension matrices perfectly.")
