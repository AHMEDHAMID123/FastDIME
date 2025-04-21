import torch
from torchvision.transforms import ToPILImage


class Diffusion:
    def __init__(
        self,
        unet,
        num_inference_steps,
        device,
        dtype,
        scheduler,
        height,
        width,
        generator: torch.Generator = None,
    ):
        self.unet = unet
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.dtype = dtype
        self.scheduler = scheduler
        self.height = height
        self.width = width
        self.generator = generator

    def __call__(self, batch_size, num_inference_steps):

        x_t = torch.randn(
            batch_size,
            3,
            self.height,
            self.width,
            device=self.device,
            dtype=self.dtype,
            generator=self.generator if self.generator is not None else None,
        )
        self.scheduler.set_inference_steps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        for t in timesteps:
            noise_pred = self.unet(x_t, t)
            x_t = self.scheduler.remove_noise(x_t, t, noise_pred)
        xt = x_t.clamp(0, 1)
        images = []
        for i in xt:
            i = ToPILImage()(i)
            images.append(i)
        return images
