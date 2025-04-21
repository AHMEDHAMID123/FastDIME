import os
import tempfile
import pytest
import torch
import yaml
from models.train_utils import (
    load_yml_file,
    TrainingConfig,
    make_grid,
    save_model,
    train_unet,
)
from PIL import Image
import numpy as np


def test_load_yml_file():
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        yaml_content = """
        test_key: test_value
        nested:
            key: value
        """
        tmp.write(yaml_content.encode())
        tmp_path = tmp.name

    try:
        config = load_yml_file(tmp_path)
        assert config["test_key"] == "test_value"
        assert config["nested"]["key"] == "value"
    finally:
        os.unlink(tmp_path)


def test_training_config():
    config = TrainingConfig()
    assert config.num_epochs == 50
    assert config.batch_size == 128
    assert config.learning_rate == 0.0001
    assert config.channels_in == 3
    assert config.channels_out == 3


def test_make_grid():
    # Create test images
    images = []
    for i in range(4):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        images.append(img)

    grid = make_grid(images, rows=2, cols=2)
    assert isinstance(grid, Image.Image)
    assert grid.size == (128, 128)  # 2x2 grid of 64x64 images


def test_save_model():
    # Create a simple model
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving model
        save_model(model, 1, optimizer, 0.5, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "epoch_1.pth"))

        # Test saving final model
        save_model(model, "final", optimizer, 0.5, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "final.pth"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_unet():
    # Create a minimal config for testing
    config = TrainingConfig(
        num_epochs=1, batch_size=2, num_training_steps=10, output_dir=tempfile.mkdtemp()
    )

    # Mock the dataset and dataloader
    class MockDataset:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.randn(3, 64, 64)

    # Replace the actual dataset with mock
    import models.train_utils as train_utils

    train_utils.CustomDataset = MockDataset

    try:
        # Run training for one epoch
        train_unet(config)

        # Check if output directory was created
        assert os.path.exists(config.output_dir)

        # Check if model checkpoint was saved
        assert os.path.exists(os.path.join(config.output_dir, "final.pth"))
    finally:
        # Cleanup
        import shutil

        shutil.rmtree(config.output_dir)


def test_training_config_validation():
    # Test invalid values
    with pytest.raises(ValueError):
        TrainingConfig(num_epochs=-1)

    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)

    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=-0.1)
