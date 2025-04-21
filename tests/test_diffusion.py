import torch
import unittest
from unittest.mock import MagicMock, patch
from models.diffusion import Diffusion


class TestDiffusion(unittest.TestCase):
    def setUp(self):
        # Create mock objects
        self.mock_unet = MagicMock()
        self.mock_scheduler = MagicMock()

        # Configure mock scheduler
        self.mock_scheduler.timesteps = torch.tensor([1, 2, 3])
        self.mock_scheduler.set_inference_steps = MagicMock()
        self.mock_scheduler.remove_noise = MagicMock(
            side_effect=lambda x, t, noise: x - 0.1
        )  # Simple mock behavior

        # Create Diffusion instance
        self.diffusion = Diffusion(
            unet=self.mock_unet,
            num_inference_steps=50,
            device="cpu",
            dtype=torch.float32,
            scheduler=self.mock_scheduler,
            height=64,
            width=64,
            generator=None,
        )

    def test_initialization(self):
        """Test that the Diffusion class initializes correctly"""
        self.assertEqual(self.diffusion.num_inference_steps, 50)
        self.assertEqual(self.diffusion.device, "cpu")
        self.assertEqual(self.diffusion.dtype, torch.float32)
        self.assertEqual(self.diffusion.height, 64)
        self.assertEqual(self.diffusion.width, 64)
        self.assertIsNone(self.diffusion.generator)

    def test_call_method(self):
        """Test the main generation functionality"""
        batch_size = 2
        num_inference_steps = 3

        # Configure mock UNet to return a simple noise prediction
        self.mock_unet.return_value = torch.zeros(batch_size, 3, 64, 64)

        # Call the diffusion model
        images = self.diffusion(batch_size, num_inference_steps)

        # Verify scheduler was called correctly
        self.mock_scheduler.set_inference_steps.assert_called_once_with(
            num_inference_steps
        )

        # Verify UNet was called for each timestep
        self.assertEqual(self.mock_unet.call_count, len(self.mock_scheduler.timesteps))

        # Verify output format
        self.assertEqual(len(images), batch_size)
        self.assertTrue(all(isinstance(img, torch.Tensor) for img in images))

        # Verify image dimensions
        for img in images:
            self.assertEqual(img.shape, (3, 64, 64))

    def test_output_clamping(self):
        """Test that the output is properly clamped between 0 and 1"""
        batch_size = 1
        num_inference_steps = 3

        # Configure mock UNet to return values that would need clamping
        self.mock_unet.return_value = (
            torch.ones(batch_size, 3, 64, 64) * 2
        )  # Values > 1

        images = self.diffusion(batch_size, num_inference_steps)

        # Verify all values are between 0 and 1
        for img in images:
            self.assertTrue(torch.all(img >= 0))
            self.assertTrue(torch.all(img <= 1))

    def test_generator_usage(self):
        """Test that the generator is used when provided"""
        generator = torch.Generator()
        diffusion = Diffusion(
            unet=self.mock_unet,
            num_inference_steps=50,
            device="cpu",
            dtype=torch.float32,
            scheduler=self.mock_scheduler,
            height=64,
            width=64,
            generator=generator,
        )

        # This is a bit tricky to test directly, but we can verify the generator is stored
        self.assertEqual(diffusion.generator, generator)


if __name__ == "__main__":
    unittest.main()
