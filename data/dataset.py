from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import pandas as pd
from PIL import Image
from pydantic import BaseModel
from typing import Union


class DataConfig(BaseModel):
    root_dir: str
    data_dir: str
    attribute_dir: str
    attribute_name: Union[list[str], None] = None
    dilimiter: Union[str, None] = None


def parse_attributes(data_directory: str, attributes: list[str]):
    data = {}
    attribute_indices = {}
    with open(data_directory, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            key = parts[0]
            values = parts[1:]

            # Determine attribute indices only once (assuming consistent order after split)
            if not attribute_indices:
                for attr in attributes:
                    try:
                        attribute_indices[attr] = values.index(attr)
                    except ValueError:
                        attribute_indices[attr] = (
                            -1
                        )  # Handle cases where attribute is missing

            extracted_values = []
            for attr in attributes:
                if attr in attribute_indices and attribute_indices[attr] != -1:
                    try:
                        extracted_values.append(values[attribute_indices[attr]])
                    except IndexError:
                        extracted_values.append(None)  # Handle potential index errors
                else:
                    extracted_values.append(None)  # Attribute not found

            data[key] = extracted_values
    return data


class CustomDataset(Dataset):
    def __init__(self, config, transform=None, mode="train"):
        self.root_dir = config.root_dir
        print(self.root_dir)
        print(config)

        self.transform = transform
        self.config = config
        self.images = os.listdir(self.config.data_dir)
        self.data_dir = self.config.data_dir
        print(self.config.attribute_dir)
        print(self.config.attribute_name)
        if self.config.attribute_dir:
            self.attributes = parse_attributes(
                self.config.attribute_dir, self.config.attribute_name
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.images[idx])
        image_name = self.images[idx]
        if self.attributes:
            attributes = self.attributes[image_name]
            if len(attributes) == 1:
                attributes = attributes[0]

        else:
            attributes = None
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, attributes if attributes is not None else image

    def visualize(self, idx_list: list[int]):
        def make_grid(images, rows, cols):
            w, h = images[0].size
            grid = Image.new("RGB", size=(cols * w, rows * h))
            for i, image in enumerate(images):
                grid.paste(image, box=(i % cols * w, i // cols * h))
            return grid

        images = []
        for idx in idx_list:
            image, attributes = self.__getitem__(idx)
            images.append(image)
        grid = make_grid(images, 1, len(idx_list))
        grid.show()
