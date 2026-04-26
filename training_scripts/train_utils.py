import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from core_engine.config import ModelConfig as cfg


def setup_logging(log_file="training_metrics.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, validation_score):
        if self.best_score is None:
            self.best_score = validation_score
            return
        if validation_score <= self.best_score + self.min_delta:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = validation_score
            self.counter = 0


class RecallWeightedFocalLoss(nn.Module):
    def __init__(self, pos_weight=1.0, gamma=2.0, ohem_ratio=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.gamma = float(gamma)
        self.ohem_ratio = float(ohem_ratio)

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = ((1.0 - p_t) ** self.gamma) * bce
        if 0 < self.ohem_ratio < 1.0:
            keep = max(1, int(focal.numel() * self.ohem_ratio))
            focal, _ = torch.topk(focal, keep)
        return focal.mean()


def composite_checkpoint_score(metrics):
    return (
        float(metrics.get("recall", 0.0)) * float(getattr(cfg, "CHECKPOINT_RECALL_WEIGHT", 0.45))
        + float(metrics.get("f1", 0.0)) * float(getattr(cfg, "CHECKPOINT_F1_WEIGHT", 0.25))
        + float(metrics.get("auc", 0.0)) * float(getattr(cfg, "CHECKPOINT_AUC_WEIGHT", 0.20))
        + float(metrics.get("precision", 0.0)) * float(getattr(cfg, "CHECKPOINT_PRECISION_WEIGHT", 0.10))
    )


class CheckpointManager:
    def __init__(self, checkpoint_dir="../models/checkpoints", task_name="generic"):
        self.checkpoint_dir = checkpoint_dir
        self.task_name = task_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_score = -1.0

    def save_checkpoint(self, model, optimizer, epoch, val_loss, metrics, threshold, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_metrics": metrics,
            "decision_threshold": threshold,
        }
        filepath = os.path.join(self.checkpoint_dir, f"authentix_{self.task_name}_epoch_{epoch}.pth")
        torch.save(checkpoint, filepath)

        score = composite_checkpoint_score(metrics)
        if is_best or score > self.best_score:
            self.best_score = score
            best_name = "authentix_best_model.pth" if self.task_name == "generic" else f"authentix_{self.task_name}_best_model.pth"
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, best_name))
            logging.info(
                f"*** New best checkpoint for {self.task_name}: score={score:.4f}, "
                f"recall={metrics.get('recall', 0.0):.4f}, f1={metrics.get('f1', 0.0):.4f}, auc={metrics.get('auc', 0.0):.4f}, threshold={threshold:.2f}"
            )


def tune_decision_threshold(y_true, y_pred_probs):
    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)
    best_threshold = 0.5
    best_score = -1.0
    precision_floor = float(getattr(cfg, "PRECISION_FLOOR", 0.45))

    for threshold in np.linspace(
        float(getattr(cfg, "THRESHOLD_SEARCH_MIN", 0.30)),
        float(getattr(cfg, "THRESHOLD_SEARCH_MAX", 0.72)),
        int(getattr(cfg, "THRESHOLD_SEARCH_STEPS", 29)),
    ):
        pred = (y_pred_probs >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        penalty = 0.0 if precision >= precision_floor else (precision_floor - precision) * 0.6
        score = (0.55 * recall) + (0.30 * f1) + (0.15 * precision) - penalty
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def calculate_metrics(y_true, y_pred_probs, threshold=None):
    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)
    threshold = float(threshold if threshold is not None else 0.5)
    y_pred_class = (y_pred_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class, zero_division=0)
    recall = recall_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "threshold": threshold,
        "checkpoint_score": composite_checkpoint_score(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
            }
        ),
    }
