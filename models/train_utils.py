from models.ddpm_schedular import DDPMSchedular
from data.dataset import CustomDataset, DataConfig
from models.unet import UNet
import torch.utils.data as DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm
from accelerate import Accelerator
from pydantic import BaseModel
import os
import torch
from PIL import Image
from models.diffusion import Diffusion
import yaml


def load_yml_file(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            for key, value in config.items():
                if isinstance(value, str):
                    if value[-5:] == ".yaml":
                        config[key] = load_yml_file(value)
            return config
        except yaml.YAMLError as exc:
            raise


class TrainingConfig(BaseModel):
    num_epochs: int = 50
    num_warmup_steps: int = 500
    save_images_epochs: int = 10
    save_model_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.0001
    eta_min: float = 0.000001
    weight_decay: float = 0.0001
    block_out_channels: tuple = (128, 128, 256, 256, 512, 512)
    num_layers_per_block: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    log_dir: str = "./logs"
    log_wuth: str = "tensorboard"
    project_name: str = "unet"
    seed: int = 42
    channels_in: int = 3
    channels_out: int = 3
    time_emb_dim: int = 128
    d_model: int = 128
    output_dir: str = "./output"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_training_steps: int = 1000
    beta_schedule: str = "linear"
    data_config: DataConfig = DataConfig(
        root_dir="./data/train",
        attributes="./data/train.csv",
        attribute_name=["image_name", "label"],
        dilimiter=",",
    )


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def save_model(model, epoch, optimizer, loss, output_dir):
    if epoch == "final":
        PATH = os.path.join(output_dir, "final.pth")
    else:
        PATH = os.path.join(output_dir, f"epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        PATH,
    )


def train_unet(config: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CustomDataset(config.data_config)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = UNet(
        time_emb_dim=None,
        in_channels=3,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        num_layers_per_block=1,
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001,
    )

    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.num_warmup_steps, eta_min=config.eta_min
    )
    ddpm_scheduler = DDPMSchedular(
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        num_training_steps=config.num_training_steps,
        beta_schedule=config.beta_schedule,
    )

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        project_name="unet",
        log_dir="./logs",
        log_wuth="tensorboard",
    )
    if accelerator.is_main_process:
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        accelerator.init_trackers(config.project_name)
    model, optimizer, lr_scheduler, train_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader
    )
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_loader), disable=not accelerator.is_main_process
        )
        progress_bar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")
        for step, images in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, config.num_training_steps, (images.shape[0],)).to(
                device
            )
            noise = torch.randn_like(images, dtype=images.dtype).to(device)

            noised_image = ddpm_scheduler.add_noise(images, noise, t)
            with accelerator.accumulate(model):
                noise_pred = model(noised_image, t)
                loss = criterion(noise_pred, noise)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            accelerator.log(logs, step=global_step)
            global_step += 1
        if (
            epoch + 1
        ) % config.save_images_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline = Diffusion(
                accelerator.unwrap_model(model),
                config.num_inference_steps,
                device,
                config.dtype,
                ddpm_scheduler,
                config.height,
                config.width,
                generator=torch.manual_seed(config.seed),
            )
            evaluate(config, epoch, pipeline)
        if (
            epoch + 1
        ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            save_model(model, epoch, optimizer, loss, config.output_dir)
