import argparse
import logging
import os
import sys

import albumentations as A
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_engine.config import ModelConfig as cfg
from core_engine.dataset_loader import build_data_loaders, build_temporal_data_loaders
from core_engine.fusion_model import AuthentixHybridModel
from core_engine.temporal_net import AuthentixTemporalLSTM
from training_scripts.train_utils import (
    CheckpointManager,
    EarlyStopping,
    RecallWeightedFocalLoss,
    calculate_metrics,
    setup_logging,
    tune_decision_threshold,
)


def _default_train_aug():
    return A.Compose(
        [
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=55, quality_upper=95, p=0.55),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=12, p=0.35),
            A.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18, hue=0.08, p=0.35),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.GaussNoise(var_limit=(10.0, 45.0), p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def _build_losses(device):
    main_loss = RecallWeightedFocalLoss(
        pos_weight=float(getattr(cfg, "FAKE_CLASS_WEIGHT", 2.6)),
        gamma=float(getattr(cfg, "FOCAL_GAMMA", 2.0)),
        ohem_ratio=float(getattr(cfg, "OHEM_KEEP_RATIO", 0.72)),
    ).to(device)
    faceswap_loss = RecallWeightedFocalLoss(
        pos_weight=float(getattr(cfg, "FACE_SWAP_CLASS_WEIGHT", 1.8)),
        gamma=float(getattr(cfg, "FOCAL_GAMMA", 2.0)),
        ohem_ratio=float(getattr(cfg, "OHEM_KEEP_RATIO", 0.72)),
    ).to(device)
    return main_loss, faceswap_loss


def train_one_epoch(model, dataloader, criteria, optimizer, device, mode="frame"):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    main_loss, faceswap_loss = criteria
    for batch in dataloader:
        optimizer.zero_grad()
        if mode == "temporal":
            image_seq, region_seq, aux_seq, labels, faceswap_labels = batch
            image_seq = image_seq.to(device)
            region_seq = region_seq.to(device)
            aux_seq = aux_seq.to(device)
            labels = labels.to(device)
            faceswap_labels = faceswap_labels.to(device)
            logits, _, faceswap_logits, _ = model(image_seq, region_seq, aux_seq)
        else:
            images, region_crops, aux_features, labels, faceswap_labels = batch
            images = images.to(device)
            region_crops = region_crops.to(device)
            aux_features = aux_features.to(device)
            labels = labels.to(device)
            faceswap_labels = faceswap_labels.to(device)
            logits, _, _, _, faceswap_logits = model(images, region_crops, aux_features)

        loss = main_loss(logits, labels) + 0.35 * faceswap_loss(faceswap_logits, faceswap_labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
        all_targets.extend(labels.detach().cpu().numpy().reshape(-1))

    threshold = tune_decision_threshold(all_targets, all_probs) if all_probs else 0.5
    metrics = calculate_metrics(all_targets, all_probs, threshold=threshold) if all_probs else calculate_metrics([0, 1], [0.0, 1.0], threshold=0.5)
    epoch_loss = running_loss / max(len(dataloader), 1)
    return epoch_loss, metrics, threshold


def validate(model, dataloader, criteria, device, mode="frame"):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    main_loss, faceswap_loss = criteria
    with torch.no_grad():
        for batch in dataloader:
            if mode == "temporal":
                image_seq, region_seq, aux_seq, labels, faceswap_labels = batch
                image_seq = image_seq.to(device)
                region_seq = region_seq.to(device)
                aux_seq = aux_seq.to(device)
                labels = labels.to(device)
                faceswap_labels = faceswap_labels.to(device)
                logits, _, faceswap_logits, _ = model(image_seq, region_seq, aux_seq)
            else:
                images, region_crops, aux_features, labels, faceswap_labels = batch
                images = images.to(device)
                region_crops = region_crops.to(device)
                aux_features = aux_features.to(device)
                labels = labels.to(device)
                faceswap_labels = faceswap_labels.to(device)
                logits, _, _, _, faceswap_logits = model(images, region_crops, aux_features)

            loss = main_loss(logits, labels) + 0.35 * faceswap_loss(faceswap_logits, faceswap_labels)
            running_loss += float(loss.item())
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            all_targets.extend(labels.cpu().numpy().reshape(-1))

    threshold = tune_decision_threshold(all_targets, all_probs) if all_probs else 0.5
    metrics = calculate_metrics(all_targets, all_probs, threshold=threshold) if all_probs else calculate_metrics([0, 1], [0.0, 1.0], threshold=0.5)
    epoch_loss = running_loss / max(len(dataloader), 1)
    return epoch_loss, metrics, threshold


def start_training_pipeline(csv_path, data_dir, task="generic"):
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    setup_logging(os.path.join(cfg.LOG_DIR, f"training_{task}.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initiating AUTHENTIX training system on {device} for task={task}")

    train_aug = _default_train_aug()
    temporal_mode = task == "temporal"
    if temporal_mode:
        train_loader, val_loader = build_temporal_data_loaders(
            csv_path=csv_path,
            data_dir=data_dir,
            batch_size=max(1, cfg.BATCH_SIZE // 4),
            train_aug=train_aug,
            num_workers=2,
            seq_length=cfg.SEQ_LENGTH,
        )
        model = AuthentixTemporalLSTM().to(device)
    else:
        train_loader, val_loader = build_data_loaders(
            csv_path=csv_path,
            data_dir=data_dir,
            batch_size=cfg.BATCH_SIZE,
            train_aug=train_aug,
            num_workers=2,
        )
        model = AuthentixHybridModel().to(device)

    criteria = _build_losses(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE)
    ckpt_manager = CheckpointManager(checkpoint_dir=cfg.CHECKPOINT_DIR, task_name=("temporal" if temporal_mode else task))

    for epoch in range(1, cfg.EPOCHS + 1):
        logging.info(f"Epoch {epoch}/{cfg.EPOCHS} started")
        train_loss, train_metrics, train_threshold = train_one_epoch(model, train_loader, criteria, optimizer, device, mode=("temporal" if temporal_mode else "frame"))
        val_loss, val_metrics, val_threshold = validate(model, val_loader, criteria, device, mode=("temporal" if temporal_mode else "frame"))

        logging.info(
            f"[TRAIN] loss={train_loss:.4f} threshold={train_threshold:.2f} recall={train_metrics['recall']:.4f} f1={train_metrics['f1']:.4f} auc={train_metrics['auc']:.4f}"
        )
        logging.info(
            f"[VAL] loss={val_loss:.4f} threshold={val_threshold:.2f} recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}"
        )

        scheduler.step()

        ckpt_manager.save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, val_threshold)
        early_stopping(val_metrics["checkpoint_score"])
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break


def main():
    parser = argparse.ArgumentParser(description="Train AUTHENTIX with recall-focused deepfake supervision.")
    parser.add_argument("--csv", required=False, default="../data/3_metadata/unified_train.csv")
    parser.add_argument("--data-dir", required=False, default="../data/2_extracted_faces")
    parser.add_argument("--task", choices=["generic", "faceswap", "temporal"], default="generic")
    args = parser.parse_args()
    start_training_pipeline(args.csv, args.data_dir, task=args.task)


if __name__ == "__main__":
    main()
