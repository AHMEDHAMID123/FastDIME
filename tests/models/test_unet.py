import torch
import pytest
from models.unet import (
    TimeEmbedding,
    residual_block,
    attention_block,
    down_block,
    up_block,
    UNet,
)


def test_time_embedding():
    """Test the TimeEmbedding module"""
    batch_size = 4
    d_embedding = 32
    time_embedding = TimeEmbedding(d_embedding)

    # Test forward pass
    time = torch.rand(batch_size)
    output = time_embedding(time)

    assert output.shape == (batch_size, d_embedding)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_residual_block():
    """Test the residual_block module"""
    batch_size = 4
    in_channels = 32
    out_channels = 64
    time_emb_dim = 128
    height, width = 32, 32

    block = residual_block(in_channels, out_channels, time_emb_dim)

    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    time_emb = torch.randn(batch_size, time_emb_dim)
    output = block(x, time_emb)

    assert output.shape == (batch_size, out_channels, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_attention_block():
    """Test the attention_block module"""
    batch_size = 4
    d_embed = 64
    num_heads = 4
    height, width = 32, 32

    block = attention_block(d_embed, num_heads)

    # Test forward pass
    x = torch.randn(batch_size, d_embed, height, width)
    output = block(x)

    assert output.shape == (batch_size, d_embed, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_down_block():
    """Test the down_block module"""
    batch_size = 4
    in_channels = 32
    out_channels = 64
    time_emb_dim = 128
    height, width = 32, 32

    block = down_block(
        in_channels,
        out_channels,
        time_emb_dim,
        layers_per_block=2,
        downsample=True,
        attention=True,
    )

    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    time_emb = torch.randn(batch_size, time_emb_dim)
    output = block(x, time_emb)

    assert output.shape == (batch_size, out_channels, height // 2, width // 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_up_block():
    """Test the up_block module"""
    batch_size = 4
    in_channels = 64
    out_channels = 32
    time_emb_dim = 128
    height, width = 16, 16

    block = up_block(
        in_channels,
        out_channels,
        time_emb_dim,
        layers_per_block=2,
        upsample=True,
        attention=True,
    )

    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    time_emb = torch.randn(batch_size, time_emb_dim)
    output = block(x, time_emb)

    assert output.shape == (batch_size, out_channels, height * 2, width * 2)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_unet():
    """Test the full UNet model"""
    batch_size = 4
    in_channels = 3
    height, width = 32, 32

    # Test with default parameters
    model = UNet()

    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width)
    time = torch.rand(batch_size)
    output = model(x, time)

    assert output.shape == (batch_size, in_channels, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Test with custom parameters
    model = UNet(
        time_emb_dim=64,
        in_channels=1,
        block_out_channels=(64, 128, 256),
        num_layers_per_block=2,
    )

    x = torch.randn(batch_size, 1, height, width)
    time = torch.rand(batch_size)
    output = model(x, time)

    assert output.shape == (batch_size, 1, height, width)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_unet_gradient_flow():
    """Test that gradients flow properly through the UNet"""
    model = UNet()
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    time = torch.rand(2)

    output = model(x, time)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
