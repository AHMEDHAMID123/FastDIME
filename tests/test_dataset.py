import os
import tempfile
import unittest
from PIL import Image
import numpy as np
from data.dataset import CustomDataset, DataConfig, parse_attributes


class TestDataset(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create test image directory
        self.image_dir = os.path.join(self.temp_dir.name, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # Create test attribute file
        self.attr_file = os.path.join(self.temp_dir.name, "attributes.txt")
        with open(self.attr_file, "w") as f:
            f.write("image1.jpg,attr1,attr2,attr3\n")
            f.write("image2.jpg,attr1,attr2,attr3\n")

        # Create test images
        for i in range(2):
            img = Image.new("RGB", (64, 64), color=(i * 100, i * 100, i * 100))
            img.save(os.path.join(self.image_dir, f"image{i+1}.jpg"))

        # Create test config
        self.config = DataConfig(
            root_dir=self.temp_dir.name,
            data_dir=self.image_dir,
            attribute_dir=self.attr_file,
            attribute_name=["attr1", "attr2", "attr3"],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_attributes(self):
        attributes = parse_attributes(self.attr_file, ["attr1", "attr2", "attr3"])

        # Check if attributes are parsed correctly
        self.assertIn("image1.jpg", attributes)
        self.assertIn("image2.jpg", attributes)
        self.assertEqual(len(attributes["image1.jpg"]), 3)
        self.assertEqual(attributes["image1.jpg"], ["attr1", "attr2", "attr3"])

    def test_dataset_length(self):
        dataset = CustomDataset(self.config)
        self.assertEqual(len(dataset), 2)

    def test_dataset_getitem(self):
        dataset = CustomDataset(self.config)

        # Test getting an item
        image, attributes = dataset[0]

        # Check image type and shape
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (64, 64))

        # Check attributes
        self.assertEqual(attributes, ["attr1", "attr2", "attr3"])

    def test_dataset_without_attributes(self):
        # Create config without attributes
        config = DataConfig(
            root_dir=self.temp_dir.name,
            data_dir=self.image_dir,
            attribute_dir=None,
            attribute_name=None,
        )

        dataset = CustomDataset(config)
        image, _ = dataset[0]

        # Check that image is returned when no attributes
        self.assertIsInstance(image, Image.Image)

    def test_dataset_with_transform(self):
        from torchvision import transforms

        # Create config with transform
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )

        dataset = CustomDataset(self.config, transform=transform)
        image, _ = dataset[0]

        # Check transformed image
        self.assertEqual(image.shape, (3, 32, 32))

    def test_visualize(self):
        dataset = CustomDataset(self.config)

        # Test visualization with a single image
        dataset.visualize([0])

        # Test visualization with multiple images
        dataset.visualize([0, 1])


if __name__ == "__main__":
    unittest.main()
