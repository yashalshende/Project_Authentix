import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_engine.dataset_loader import AuthentixUnifiedDataset
from core_engine.fusion_model import AuthentixHybridModel
from training_scripts.eval_utils import evaluate_predictions, plot_confusion_matrix, save_evaluation_results


def run_standard_evaluation(test_csv, model_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = AuthentixHybridModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    base_threshold = float(checkpoint.get("decision_threshold", 0.5))

    df = pd.read_csv(test_csv)
    dataset_sources = df["dataset_source"].unique().tolist()
    all_results = []
    global_targets = []
    global_probs = []
    base_dir = os.path.dirname(os.path.dirname(test_csv))

    for ds in dataset_sources:
        ds_df = df[(df["dataset_source"] == ds) & (df["split"] == "test")].reset_index(drop=True)
        temp_csv = f"temp_eval_{ds}.csv"
        ds_df.to_csv(temp_csv, index=False)

        dataset = AuthentixUnifiedDataset(temp_csv, root_dir=base_dir, split="test", transform=None)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        targets, probs = [], []
        with torch.no_grad():
            for images, region_crops, aux_features, labels, _ in tqdm(loader, desc=f"Scanning {ds}"):
                images = images.to(device)
                region_crops = region_crops.to(device)
                aux_features = aux_features.to(device)
                logits, _, _, _, _ = model(images, region_crops, aux_features)
                probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                targets.extend(labels.numpy().flatten())

        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        if probs:
            res = evaluate_predictions(targets, probs, ds, threshold=base_threshold)
            plot_confusion_matrix(res["confusion_matrix"], ds)
            all_results.append(res)
            global_targets.extend(targets)
            global_probs.extend(probs)

    if global_probs:
        combined_res = evaluate_predictions(global_targets, global_probs, "Combined_Test_Set", threshold=base_threshold)
        plot_confusion_matrix(combined_res["confusion_matrix"], "Combined_Test_Set")
        all_results.append(combined_res)
        save_evaluation_results(all_results, filename="standard_eval_metrics.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../data/3_metadata/unified_test.csv")
    parser.add_argument("--model-path", default="../models/checkpoints/authentix_best_model.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--task", choices=["generic", "faceswap"], default="generic")
    args = parser.parse_args()
    run_standard_evaluation(args.csv, args.model_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
