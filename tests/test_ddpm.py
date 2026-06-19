import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from ddpm_diffusion import Diffusion, make_beta_schedule
from ddpm_common import load_checkpoint, save_checkpoint
from ddpm_models import SimpleUNet


def test_unconditional_unet_shape():
    model = SimpleUNet(image_channels=1, base_channels=8)
    images = torch.randn(2, 1, 32, 32)
    timesteps = torch.tensor([0, 9])
    assert model(images, timesteps).shape == images.shape


def test_conditional_unet_and_null_label_shape():
    model = SimpleUNet(image_channels=1, base_channels=8, num_classes=10)
    images = torch.randn(2, 1, 32, 32)
    timesteps = torch.tensor([1, 7])
    labels = torch.tensor([3, model.null_label])
    assert model(images, timesteps, labels).shape == images.shape
    assert model(images, timesteps, None).shape == images.shape


@pytest.mark.parametrize("name", ["linear", "cosine"])
def test_schedules_are_valid(name):
    betas = make_beta_schedule(name, 20)
    assert betas.shape == (20,)
    assert torch.isfinite(betas).all()
    assert (betas > 0).all() and (betas < 1).all()


def test_q_sample_shape_and_finite_values():
    diffusion = Diffusion(timesteps=20)
    images = torch.randn(4, 1, 32, 32)
    times = torch.tensor([0, 1, 10, 19])
    noisy = diffusion.q_sample(images, times)
    assert noisy.shape == images.shape
    assert torch.isfinite(noisy).all()


def test_short_ddim_sampling():
    class ZeroNoise(torch.nn.Module):
        num_classes = None

        def forward(self, x, timesteps, labels=None):
            return torch.zeros_like(x)

    diffusion = Diffusion(timesteps=10)
    samples, frames = diffusion.sample_ddim(
        ZeroNoise(), (2, 1, 8, 8), sampling_steps=5, return_frames=True
    )
    assert samples.shape == (2, 1, 8, 8)
    assert frames
    assert torch.isfinite(samples).all()


def test_checkpoint_round_trip(tmp_path):
    state = {"epoch": 3, "model": {"weight": torch.tensor([1.0, 2.0])}}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(state, path)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert loaded["epoch"] == 3
    assert torch.equal(loaded["model"]["weight"], state["model"]["weight"])

