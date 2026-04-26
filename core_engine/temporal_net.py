import torch
import torch.nn as nn

try:
    from core_engine.config import ModelConfig as cfg
    from core_engine.fusion_model import AuthentixHybridModel
except ImportError:
    from config import ModelConfig as cfg
    from fusion_model import AuthentixHybridModel


class AuthentixTemporalLSTM(nn.Module):
    def __init__(self, pretrained_hybrid=None):
        super().__init__()
        self.base_model = pretrained_hybrid if pretrained_hybrid is not None else AuthentixHybridModel()
        self.base_model.return_embedding = True

        self.lstm = nn.LSTM(
            input_size=cfg.FUSION_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            dropout=0.0 if cfg.LSTM_LAYERS == 1 else cfg.DROPOUT_RATE,
        )
        self.temporal_classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE),
            nn.Linear(cfg.LSTM_HIDDEN, cfg.LSTM_HIDDEN // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.LSTM_HIDDEN // 2, cfg.NUM_CLASSES),
        )
        self.temporal_faceswap_head = nn.Sequential(
            nn.Dropout(cfg.DROPOUT_RATE * 0.8),
            nn.Linear(cfg.LSTM_HIDDEN, cfg.LSTM_HIDDEN // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.LSTM_HIDDEN // 2, 1),
        )

    def forward(self, image_sequence, region_sequence=None, aux_sequence=None):
        batch_size, seq_len, channels, height, width = image_sequence.size()
        flat_images = image_sequence.view(batch_size * seq_len, channels, height, width)

        flat_regions = None
        if region_sequence is not None:
            _, _, num_regions, region_channels, region_height, region_width = region_sequence.size()
            flat_regions = region_sequence.view(batch_size * seq_len, num_regions, region_channels, region_height, region_width)

        flat_aux = None
        if aux_sequence is not None:
            flat_aux = aux_sequence.view(batch_size * seq_len, -1)

        _, spatial_maps, embeddings, _, _, frame_faceswap_logits = self.base_model(flat_images, flat_regions, flat_aux)
        lstm_input = embeddings.view(batch_size, seq_len, -1)
        _, (hidden_state, _) = self.lstm(lstm_input)
        video_context_vector = hidden_state[-1]

        video_verdict_logit = self.temporal_classifier(video_context_vector)
        video_faceswap_logit = self.temporal_faceswap_head(video_context_vector)

        center_frame_index = seq_len // 2
        spatial_maps_resolved = spatial_maps.view(
            batch_size,
            seq_len,
            spatial_maps.size(1),
            spatial_maps.size(2),
            spatial_maps.size(3),
        )
        center_cam_map = spatial_maps_resolved[:, center_frame_index, :, :, :]
        frame_faceswap_logits = frame_faceswap_logits.view(batch_size, seq_len, -1)
        return video_verdict_logit, center_cam_map, video_faceswap_logit, frame_faceswap_logits


if __name__ == "__main__":
    print("Initiating AUTHENTIX Temporal Network LSTM Verification...")
    video_tensor = torch.randn(2, cfg.SEQ_LENGTH, 3, 256, 256)
    region_tensor = torch.randn(2, cfg.SEQ_LENGTH, cfg.NUM_FACE_REGIONS, 3, cfg.REGION_SIZE, cfg.REGION_SIZE)
    aux_tensor = torch.randn(2, cfg.SEQ_LENGTH, cfg.FACE_SWAP_AUX_DIM)
    temporal_model = AuthentixTemporalLSTM()

    video_logit, cam_map, video_faceswap_logit, frame_faceswap_logits = temporal_model(video_tensor, region_tensor, aux_tensor)
    print(f"Ultimate Video Logits Matrix Shape: {video_logit.shape} (Anticipated: 2, 1)")
    print(f"Center Timeline XAI Mapping Frame Array: {cam_map.shape} (Anticipated: 2, 1280, 8, 8)")
    print(f"Video Face Swap Logits Shape: {video_faceswap_logit.shape} (Anticipated: 2, 1)")
    print(f"Frame Face Swap Logits Shape: {frame_faceswap_logits.shape} (Anticipated: 2, {cfg.SEQ_LENGTH}, 1)")
