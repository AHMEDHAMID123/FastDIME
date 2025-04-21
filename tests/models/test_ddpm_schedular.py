import torch
import pytest
from models.ddpm_schedular import DDPMSchedular


def test_ddpm_scheduler_initialization():
    # Test default initialization
    scheduler = DDPMSchedular()
    assert scheduler.beta_start == 0.0001
    assert scheduler.beta_end == 0.02
    assert scheduler.num_training_steps == 1000
    assert scheduler.beta_schedule == "linear"
    assert len(scheduler.betas) == 1000
    assert len(scheduler.alphas) == 1000
    assert len(scheduler.alpha_bars) == 1000

    # Test custom initialization
    scheduler = DDPMSchedular(
        beta_start=0.001, beta_end=0.1, num_training_steps=500, beta_schedule="cosine"
    )
    assert scheduler.beta_start == 0.001
    assert scheduler.beta_end == 0.1
    assert scheduler.num_training_steps == 500
    assert scheduler.beta_schedule == "cosine"


def test_beta_schedules():
    # Test linear schedule
    scheduler = DDPMSchedular(beta_schedule="linear")
    betas = scheduler.betas
    assert torch.allclose(betas[0], torch.tensor(0.0001))
    assert torch.allclose(betas[-1], torch.tensor(0.02))
    assert torch.all(betas[:-1] <= betas[1:])  # Monotonically increasing

    # Test cosine schedule
    scheduler = DDPMSchedular(beta_schedule="cosine")
    betas = scheduler.betas
    assert torch.allclose(betas[0], torch.tensor(1.0))
    assert torch.allclose(betas[-1], torch.tensor(-1.0))
    assert torch.all(betas[:-1] >= betas[1:])  # Monotonically decreasing


def test_inference_steps():
    scheduler = DDPMSchedular()
    scheduler.set_inference_steps(100)
    assert scheduler.num_inference_steps == 100
    assert len(scheduler.timesteps) == 100
    assert torch.all(scheduler.timesteps[:-1] >= scheduler.timesteps[1:])  # Decreasing


def test_add_noise():
    scheduler = DDPMSchedular()
    batch_size = 2
    channels = 3
    height = 32
    width = 32

    # Create test tensors
    original_x = torch.ones((batch_size, channels, height, width))
    t = torch.tensor([0, 500])  # Test different timesteps

    # Test noise addition
    noisy_x = scheduler.add_noise(original_x, None, t)

    assert noisy_x.shape == original_x.shape
    assert not torch.allclose(noisy_x, original_x)  # Should be different due to noise
    assert torch.allclose(
        noisy_x.mean(), original_x.mean(), rtol=0.1
    )  # Mean should be similar


def test_remove_noise():
    scheduler = DDPMSchedular()
    batch_size = 2
    channels = 3
    height = 32
    width = 32

    # Create test tensors
    xt = torch.randn((batch_size, channels, height, width))
    t = torch.tensor([0, 500])
    model_output = torch.randn((batch_size, channels, height, width))

    # Test noise removal
    x_t_minus_1 = scheduler.remove_noise(xt, t, model_output)

    assert x_t_minus_1.shape == xt.shape
    assert not torch.allclose(x_t_minus_1, xt)  # Should be different


def test_invalid_beta_schedule():
    with pytest.raises(ValueError):
        DDPMSchedular(beta_schedule="invalid")


def test_device_handling():
    # Test that tensors are moved to correct device
    scheduler = DDPMSchedular()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_x = torch.ones((2, 3, 32, 32), device=device)
    t = torch.tensor([0, 500], device=device)

    noisy_x = scheduler.add_noise(original_x, None, t)
    assert noisy_x.device == device
