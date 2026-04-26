import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from core_engine.config import ModelConfig as cfg
from training_scripts.train_utils import tune_decision_threshold


def evaluate_predictions(y_true, y_pred_probs, dataset_name, threshold=None):
    y_true = np.array(y_true).astype(int)
    y_pred_probs = np.array(y_pred_probs, dtype=np.float32)
    threshold = float(threshold if threshold is not None else tune_decision_threshold(y_true, y_pred_probs))
    y_pred_class = (y_pred_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class, zero_division=0)
    recall = recall_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred_class).tolist()
    return {
        "dataset": dataset_name,
        "threshold": threshold,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        },
        "confusion_matrix": cm,
    }


def save_evaluation_results(results, output_dir="../models/evaluations", filename="eval_results.json"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    csv_path = out_path.replace(".json", ".csv")
    flattened = []
    iterable = results if isinstance(results, list) else [results]
    for row in iterable:
        flattened.append({"dataset": row["dataset"], "threshold": row.get("threshold", 0.5), **row["metrics"]})
    pd.DataFrame(flattened).to_csv(csv_path, index=False)
    print(f"Metrics saved to: {csv_path}")


def plot_confusion_matrix(cm, dataset_name, output_dir="../models/evaluations"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual True Class")
    plt.title(f"Forensic Confusion Matrix - {dataset_name}")
    plt.tight_layout()
    outname = dataset_name.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(output_dir, f"cm_{outname}.png"))
    plt.close()
