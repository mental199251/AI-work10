"""Fail-fast environment and model smoke check for a remote server."""

from __future__ import annotations

import argparse

import torch
import torchvision
from PIL import Image

from ddpm_common import environment_info, get_device
from ddpm_diffusion import Diffusion
from ddpm_models import SimpleUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = get_device(args.device)
    print(environment_info())
    print(f"torchvision: {torchvision.__version__}")
    print(f"Pillow: {Image.__version__ if hasattr(Image, '__version__') else 'installed'}")
    print(f"Selected device: {device}")
    model = SimpleUNet(image_channels=1, base_channels=8, num_classes=10).to(device)
    images = torch.randn(2, 1, 32, 32, device=device)
    timesteps = torch.tensor([0, 9], device=device)
    labels = torch.tensor([1, model.null_label], device=device)
    output = model(images, timesteps, labels)
    if output.shape != images.shape or not torch.isfinite(output).all():
        raise RuntimeError("Model forward smoke check failed.")
    for name in ("linear", "cosine"):
        diffusion = Diffusion(timesteps=20, beta_schedule=name, device=str(device))
        noisy = diffusion.q_sample(images, timesteps)
        if not torch.isfinite(noisy).all():
            raise RuntimeError(f"{name} diffusion smoke check failed.")
    print("Environment and tensor smoke checks passed.")


if __name__ == "__main__":
    main()
