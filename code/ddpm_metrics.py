"""MNIST classifier and quantitative metrics for conditional samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ddpm_common import load_checkpoint


class MNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = self.features(images)
        flattened = features.flatten(1)
        if return_features:
            return flattened
        return self.classifier(features)


def load_classifier(path: Path, device: torch.device) -> MNISTClassifier:
    checkpoint = load_checkpoint(path, device)
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(checkpoint.get("model", checkpoint))
    classifier.eval()
    return classifier


@torch.no_grad()
def evaluate_conditional_samples(
    classifier: MNISTClassifier,
    images: torch.Tensor,
    target_labels: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    images = images.to(device)
    target_labels = target_labels.to(device)
    probabilities = classifier(images).softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for target, prediction in zip(target_labels.cpu(), predictions.cpu()):
        confusion[int(target), int(prediction)] += 1
    metrics: Dict[str, Any] = {
        "conditional_accuracy": float((predictions == target_labels).float().mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "pixel_diversity": float(images.flatten(1).std(dim=0).mean().item()),
        "sample_count": int(images.shape[0]),
    }
    for digit in range(10):
        mask = target_labels == digit
        metrics[f"accuracy_digit_{digit}"] = float(
            (predictions[mask] == digit).float().mean().item()
        ) if mask.any() else float("nan")
    return metrics, confusion


def save_confusion_matrix(matrix: torch.Tensor, path: Path, title: str) -> None:
    array = matrix.cpu().numpy()
    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    image = axis.imshow(array, cmap="Blues")
    for row in range(10):
        for col in range(10):
            axis.text(col, row, str(array[row, col]), ha="center", va="center", fontsize=7)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Target label")
    axis.set_title(title)
    axis.set_xticks(np.arange(10))
    axis.set_yticks(np.arange(10))
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
