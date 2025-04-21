import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DDPMSchedular:
    def __init__(
        self,
        beta_start=0.0001,
        beta_end=0.02,
        num_training_steps=1000,
        beta_schedule="linear",
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_training_steps = num_training_steps
        self.beta_schedule = beta_schedule
        self.timesteps = torch.arange(0, num_training_steps).flip(0)
        self.betas = self.get_betas(
            self.beta_start, self.beta_end, self.num_training_steps, self.beta_schedule
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def get_betas(self, beta_start, beta_end, num_training_steps, beta_schedule):
        if beta_schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_training_steps)
        elif beta_schedule == "cosine":
            return torch.cos(torch.linspace(0, math.pi, num_training_steps))
        else:
            raise ValueError(
                f"Unsupported beta schedule: {beta_schedule}, please use 'linear' or 'cosine'"
            )

    def set_inference_steps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        step = self.num_training_steps // self.num_inference_steps
        self.timesteps = torch.linspace(0, self.num_training_steps, step).flip(0)

    def add_noise(
        self, original_x: torch.Tensor, noise: torch.Tensor, t: torch.IntTensor
    ):

        sqrt_alpha_bars = torch.sqrt(self.alpha_bars[t]).to(original_x.device)
        while len(sqrt_alpha_bars.shape) < len(original_x.shape):
            sqrt_alpha_bars = sqrt_alpha_bars.unsqueeze(-1)
        sqrt_one_minus_alpha_bars = torch.sqrt(
            1 - self.alpha_bars[t],
        ).to(original_x.device)
        noise = torch.randn_like(original_x, dtype=original_x.dtype).to(
            original_x.device
        )
        print(sqrt_alpha_bars, original_x.shape, sqrt_one_minus_alpha_bars, noise.shape)
        return sqrt_alpha_bars * original_x + sqrt_one_minus_alpha_bars * noise

    def remove_noise(
        self, xt: torch.Tensor, t: torch.IntTensor, model_output: torch.Tensor
    ):
        alpha_t = self.alphas[t]
        sqrt_alpha_bars = torch.sqrt(self.alpha_bars[t], dtype=xt.dtype).to(xt.device)
        while len(sqrt_alpha_bars.shape) < len(xt.shape):
            sqrt_alpha_bars = sqrt_alpha_bars.unsqueeze(-1)
        sqrt_one_minus_alpha_bars = torch.sqrt(
            1 - self.alpha_bars[t], dtype=xt.dtype
        ).to(xt.device)
        beta_t = 1 - alpha_t
        z = torch.randn_like(xt, dtype=xt.dtype).to(xt.device)
        x_t_minus_1 = (
            xt - (1 - alpha_t) / (sqrt_one_minus_alpha_bars) * model_output
        ) / alpha_t + beta_t * z
        return x_t_minus_1
