"""Dataset loading and safe preparation helpers."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from ddpm_common import ensure_dir


def subset_dataset(dataset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def prepare_cifar10_data(data_dir: Path, archive: Optional[Path] = None) -> Path:
    data_dir = ensure_dir(data_dir)
    extracted = data_dir / "cifar-10-batches-py"
    if extracted.exists():
        return extracted
    candidates = [data_dir / "cifar-10-python.tar.gz"]
    if archive is not None:
        candidates.insert(0, Path(archive))
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "CIFAR-10 was not found. Put cifar-10-python.tar.gz in the data directory "
            "or pass --cifar-archive."
        )
    data_root = data_dir.resolve()
    with tarfile.open(source, "r:gz") as tar:
        for member in tar.getmembers():
            target = (data_dir / member.name).resolve()
            try:
                target.relative_to(data_root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe archive path: {member.name}") from exc
        tar.extractall(data_dir)
    if not extracted.exists():
        raise RuntimeError("The CIFAR-10 archive did not contain cifar-10-batches-py.")
    return extracted


def get_dataset(
    name: str,
    data_dir: Path,
    train: bool,
    max_samples: Optional[int] = None,
    seed: int = 42,
    download: bool = True,
    cifar_archive: Optional[Path] = None,
) -> Tuple[Dataset, int, int]:
    name = name.lower()
    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        dataset = datasets.MNIST(
            root=data_dir, train=train, download=download, transform=transform
        )
        return subset_dataset(dataset, max_samples, seed), 1, 10
    if name == "cifar10":
        prepare_cifar10_data(data_dir, cifar_archive)
        transform_steps = []
        if train:
            transform_steps.append(transforms.RandomHorizontalFlip())
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=False,
            transform=transforms.Compose(transform_steps),
        )
        return subset_dataset(dataset, max_samples, seed), 3, 10
    raise ValueError(f"Unknown dataset: {name}")

