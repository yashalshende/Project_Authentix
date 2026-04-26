import os
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from core_engine.config import ModelConfig as cfg
    from utils.face_regions import FaceRegionAnalyzer, REGION_LAYOUT
except ImportError:
    from config import ModelConfig as cfg
    from utils.face_regions import FaceRegionAnalyzer, REGION_LAYOUT


AUX_COLUMNS = [
    "identity_dispersion",
    "left_right_identity_gap",
    "center_periphery_gap",
    "landmark_eye_alignment",
    "landmark_mouth_geometry",
    "landmark_face_symmetry",
    "landmark_contour_consistency",
    "boundary_anomaly_score",
    "texture_mismatch_score",
    "face_quality_score",
]


class AuthentixUnifiedDataset(Dataset):
    def __init__(self, csv_file, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        full_df = pd.read_csv(csv_file)
        self.data_df = full_df[full_df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.face_region_analyzer = FaceRegionAnalyzer()

        self.default_transform = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.region_transform = A.Compose(
            [
                A.Resize(cfg.REGION_SIZE, cfg.REGION_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_df.iloc[idx]
        img_name = os.path.join(self.root_dir, row["file_path"])
        label = int(row.get("label", 0))
        faceswap_label = int(row.get("faceswap_label", label))

        raw_image = cv2.imread(img_name)
        if raw_image is None:
            raw_image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image = raw_image.copy()

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = self.default_transform(image=image)["image"]

        region_tensor = self._load_region_tensor(row, raw_image)
        aux_tensor = self._load_aux_features(row)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        faceswap_tensor = torch.tensor([faceswap_label], dtype=torch.float32)
        return image, region_tensor, aux_tensor, label_tensor, faceswap_tensor

    def _load_region_tensor(self, row, raw_image):
        region_dir = row.get("region_dir", "")
        region_tensors = []

        if isinstance(region_dir, str) and region_dir:
            region_dir_abs = os.path.join(self.root_dir, region_dir)
            for region_key, _ in REGION_LAYOUT:
                region_path = os.path.join(region_dir_abs, f"{region_key}.jpg")
                region_img = cv2.imread(region_path)
                if region_img is None:
                    region_img = np.zeros((cfg.REGION_SIZE, cfg.REGION_SIZE, 3), dtype=np.uint8)
                region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
                region_tensors.append(self.region_transform(image=region_img)["image"])
        else:
            region_stack = self.face_region_analyzer.extract_region_tensor_stack(raw_image, output_size=cfg.REGION_SIZE)
            for region_img in region_stack:
                region_tensors.append(self.region_transform(image=region_img)["image"])

        return torch.stack(region_tensors, dim=0)

    def _load_aux_features(self, row):
        values = []
        for column in AUX_COLUMNS:
            values.append(float(row.get(column, 0.0) or 0.0))
        values = np.asarray(values, dtype=np.float32)
        target_dim = int(getattr(cfg, "FACE_SWAP_AUX_DIM", len(AUX_COLUMNS)))
        if values.shape[0] < target_dim:
            values = np.pad(values, (0, target_dim - values.shape[0]))
        return torch.tensor(values[:target_dim], dtype=torch.float32)


class AuthentixTemporalDataset(Dataset):
    def __init__(self, csv_file, root_dir, split="train", transform=None, seq_length=None):
        self.root_dir = root_dir
        self.transform = transform
        self.seq_length = int(seq_length or getattr(cfg, "SEQ_LENGTH", 10))
        self.face_region_analyzer = FaceRegionAnalyzer()

        full_df = pd.read_csv(csv_file)
        split_df = full_df[full_df["split"] == split].reset_index(drop=True)
        key_to_rows = defaultdict(list)
        for _, row in split_df.iterrows():
            key = (row.get("dataset_source", "UNK"), row.get("parent_video_id", row.get("file_path")))
            key_to_rows[key].append(row)

        self.samples = []
        for _, rows in key_to_rows.items():
            rows = sorted(rows, key=lambda item: int(item.get("frame_idx", 0)))
            self.samples.append(rows)

        self.default_transform = A.Compose(
            [
                A.Resize(256, 256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.region_transform = A.Compose(
            [
                A.Resize(cfg.REGION_SIZE, cfg.REGION_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rows = self.samples[idx]
        rows = self._sample_rows(rows)

        image_seq = []
        region_seq = []
        aux_seq = []
        label = int(rows[0].get("label", 0))
        faceswap_label = int(rows[0].get("faceswap_label", label))

        for row in rows:
            raw_image = cv2.imread(os.path.join(self.root_dir, row["file_path"]))
            if raw_image is None:
                raw_image = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image_tensor = self.transform(image=raw_image)["image"]
            else:
                image_tensor = self.default_transform(image=raw_image)["image"]
            image_seq.append(image_tensor)
            region_seq.append(self._load_region_tensor(row, raw_image))
            aux_seq.append(self._load_aux_features(row))

        return (
            torch.stack(image_seq, dim=0),
            torch.stack(region_seq, dim=0),
            torch.stack(aux_seq, dim=0),
            torch.tensor([label], dtype=torch.float32),
            torch.tensor([faceswap_label], dtype=torch.float32),
        )

    def _sample_rows(self, rows):
        if len(rows) >= self.seq_length:
            indices = np.linspace(0, len(rows) - 1, self.seq_length, dtype=int)
            return [rows[index] for index in indices]
        padded = list(rows)
        while len(padded) < self.seq_length:
            padded.append(rows[-1])
        return padded

    def _load_region_tensor(self, row, raw_image):
        region_dir = row.get("region_dir", "")
        region_tensors = []
        if isinstance(region_dir, str) and region_dir:
            region_dir_abs = os.path.join(self.root_dir, region_dir)
            for region_key, _ in REGION_LAYOUT:
                region_path = os.path.join(region_dir_abs, f"{region_key}.jpg")
                region_img = cv2.imread(region_path)
                if region_img is None:
                    region_img = np.zeros((cfg.REGION_SIZE, cfg.REGION_SIZE, 3), dtype=np.uint8)
                region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
                region_tensors.append(self.region_transform(image=region_img)["image"])
        else:
            region_stack = self.face_region_analyzer.extract_region_tensor_stack(raw_image, output_size=cfg.REGION_SIZE)
            for region_img in region_stack:
                region_tensors.append(self.region_transform(image=region_img)["image"])
        return torch.stack(region_tensors, dim=0)

    def _load_aux_features(self, row):
        values = np.asarray([float(row.get(column, 0.0) or 0.0) for column in AUX_COLUMNS], dtype=np.float32)
        target_dim = int(getattr(cfg, "FACE_SWAP_AUX_DIM", len(AUX_COLUMNS)))
        if values.shape[0] < target_dim:
            values = np.pad(values, (0, target_dim - values.shape[0]))
        return torch.tensor(values[:target_dim], dtype=torch.float32)


def _build_sample_weights(df):
    weights = []
    for _, row in df.iterrows():
        weight = 1.0
        label = int(row.get("label", 0))
        faceswap_label = int(row.get("faceswap_label", 0))
        manipulation = str(row.get("manipulation_type", "")).lower()

        if label == 1:
            weight *= float(getattr(cfg, "HARD_FAKE_OVERSAMPLE", 2.8))
        if faceswap_label == 1 or manipulation == "faceswap":
            weight *= float(getattr(cfg, "FACE_SWAP_OVERSAMPLE", 1.6))
        if float(row.get("boundary_anomaly_score", 0.0) or 0.0) >= float(getattr(cfg, "HARD_SAMPLE_SCORE", 55.0)):
            weight *= 1.2
        if float(row.get("identity_dispersion", 0.0) or 0.0) >= 18.0:
            weight *= 1.15
        weights.append(weight)
    return np.asarray(weights, dtype=np.float64)


def build_data_loaders(csv_path, data_dir, batch_size=16, train_aug=None, num_workers=4):
    train_dataset = AuthentixUnifiedDataset(csv_path, data_dir, split="train", transform=train_aug)
    val_dataset = AuthentixUnifiedDataset(csv_path, data_dir, split="val", transform=None)

    sampler = None
    if len(train_dataset.data_df) > 0:
        sample_weights = _build_sample_weights(train_dataset.data_df)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_temporal_data_loaders(csv_path, data_dir, batch_size=4, train_aug=None, num_workers=2, seq_length=None):
    train_dataset = AuthentixTemporalDataset(csv_path, data_dir, split="train", transform=train_aug, seq_length=seq_length)
    val_dataset = AuthentixTemporalDataset(csv_path, data_dir, split="val", transform=None, seq_length=seq_length)

    sampler = None
    if len(train_dataset.samples) > 0:
        weights = []
        for rows in train_dataset.samples:
            row = rows[0]
            weight = 1.0
            if int(row.get("label", 0)) == 1:
                weight *= float(getattr(cfg, "HARD_FAKE_OVERSAMPLE", 2.8))
            if int(row.get("faceswap_label", 0)) == 1:
                weight *= float(getattr(cfg, "FACE_SWAP_OVERSAMPLE", 1.6))
            weights.append(weight)
        sampler = WeightedRandomSampler(np.asarray(weights, dtype=np.float64), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
