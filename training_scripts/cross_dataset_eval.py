import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_engine.dataset_loader import AuthentixUnifiedDataset
from core_engine.fusion_model import AuthentixHybridModel
from training_scripts.eval_utils import evaluate_predictions, plot_confusion_matrix, save_evaluation_results


def run_cross_dataset_evaluation(cross_csv, model_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = AuthentixHybridModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(checkpoint.get("decision_threshold", 0.5))

    base_dir = os.path.dirname(os.path.dirname(cross_csv))
    dataset = AuthentixUnifiedDataset(cross_csv, root_dir=base_dir, split="test", transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    targets, probs = [], []
    with torch.no_grad():
        for images, region_crops, aux_features, labels, _ in tqdm(loader, desc="Scanning Alien Datasets"):
            images = images.to(device)
            region_crops = region_crops.to(device)
            aux_features = aux_features.to(device)
            logits, _, _, _, _ = model(images, region_crops, aux_features)
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            targets.extend(labels.numpy().flatten())

    if probs:
        res = evaluate_predictions(targets, probs, "Cross_Dataset_Generalization", threshold=threshold)
        plot_confusion_matrix(res["confusion_matrix"], "Cross_Dataset_Generalization")
        save_evaluation_results([res], filename="cross_dataset_robustness_metrics.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../data/3_metadata/cross_dataset_eval.csv")
    parser.add_argument("--model-path", default="../models/checkpoints/authentix_best_model.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run_cross_dataset_evaluation(args.csv, args.model_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
