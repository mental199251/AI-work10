"""Train the independent MNIST evaluator used for conditional generation metrics."""

from __future__ import annotations

import argparse
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ddpm_common import ensure_dir, get_device, save_checkpoint, save_json, set_seed
from ddpm_data import get_dataset
from ddpm_metrics import MNISTClassifier
from project_cli import project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/mnist_classifier")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = ensure_dir(project_path(args.output_dir))
    train_set, _, _ = get_dataset("mnist", project_path(args.data_dir), True, seed=args.seed)
    test_set, _, _ = get_dataset("mnist", project_path(args.data_dir), False, seed=args.seed)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = MNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]
        print(f"epoch {epoch}/{args.epochs} | loss {running_loss / len(train_set):.6f}")
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum())
    accuracy = correct / len(test_set)
    save_checkpoint({"model": model.state_dict(), "test_accuracy": accuracy}, output_dir / "classifier.pt")
    save_json(
        {"test_accuracy": accuracy, "training_seconds": time.perf_counter() - started},
        output_dir / "metrics.json",
    )
    print(f"Test accuracy: {accuracy:.4%}")


if __name__ == "__main__":
    main()

