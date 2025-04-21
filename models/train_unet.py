import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from argparse import ArgumentParser
from models.train_utils import load_yml_file, TrainingConfig, train_unet


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.yaml")
    args = parser.parse_args()
    config = load_yml_file(args.config)
    train_unet(config)


if __name__ == "__main__":
    main()
