import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ddpm_common import (
    DiffusionSchedule,
    SimpleUNet,
    count_parameters,
    ensure_dir,
    get_device,
    prepare_cifar10_data,
    save_checkpoint,
    save_tensor_grid,
    set_seed,
    subset_dataset,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "cifar10"

EPOCHS = 30
BATCH_SIZE = 96
LEARNING_RATE = 2e-4
TIMESTEPS = 300
BASE_CHANNELS = 48
MAX_TRAIN_SAMPLES = 10000
NUM_WORKERS = 0
SAMPLE_EVERY_EPOCH = True
DEVICE = "auto"
SEED = 42


def main():
    set_seed(SEED)
    device = get_device(DEVICE)
    output_dir = ensure_dir(OUTPUT_DIR)
    prepare_cifar10_data(DATA_DIR)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_set = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform,
    )
    train_set = subset_dataset(train_set, MAX_TRAIN_SAMPLES)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    model = SimpleUNet(image_channels=3, base_channels=BASE_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    diffusion = DiffusionSchedule(timesteps=TIMESTEPS, device=device)

    config = {
        "dataset": "CIFAR10",
        "image_size": 32,
        "image_channels": 3,
        "base_channels": BASE_CHANNELS,
        "timesteps": TIMESTEPS,
        "beta_start": 1e-4,
        "beta_end": 0.02,
    }

    print("Optional task: train CIFAR-10 color DDPM")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_set)}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Output directory: {output_dir}")

    global_step = 0
    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            batch_size = images.shape[0]
            t = torch.randint(0, TIMESTEPS, (batch_size,), device=device).long()
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, t, noise)
            predicted_noise = model(noisy_images, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                avg_loss = running_loss / batch_idx
                print(f"epoch {epoch}/{EPOCHS} | batch {batch_idx}/{len(train_loader)} | loss {avg_loss:.4f}")

        latest_path = output_dir / "cifar10_ddpm_latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch, global_step, config)
        save_checkpoint(output_dir / f"cifar10_ddpm_epoch_{epoch}.pt", model, optimizer, epoch, global_step, config)

        if SAMPLE_EVERY_EPOCH:
            samples, _ = diffusion.sample(model, shape=(16, 3, 32, 32), save_every=50, return_frames=False)
            save_tensor_grid(samples, output_dir / f"sample_epoch_{epoch}.png", nrow=4)

    elapsed = time.time() - start_time
    print(f"Done. Latest checkpoint: {output_dir / 'cifar10_ddpm_latest.pt'}")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
