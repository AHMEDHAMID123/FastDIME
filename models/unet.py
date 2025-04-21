import torch
import torch.nn as nn
import torch.functional as F
import math


class TimeEmbedding(nn.Module):
    def __init__(self, d_embedding, freq_scale=10000):
        super().__init__()
        self.d_embedding = d_embedding
        self.freq_scale = freq_scale
        self.linear1 = nn.Linear(d_embedding, d_embedding * 4)
        self.linear2 = nn.Linear(d_embedding * 4, d_embedding)
        self.act = nn.SiLU()

    def forward(self, time):
        time_embedding = self.get_time_embedding(time)
        time_embedding = self.linear1(time_embedding)
        time_embedding = self.act(time_embedding)
        time_embedding = self.linear2(time_embedding)

        return time_embedding

    def get_time_embedding(self, t):
        half_dim = self.d_embedding // 2
        # sinusoidal embedding similar to that used in transformer
        # sin(t * w_k^-(i/(d/2))) + cos(t * w_k^-(i/(d/2)))
        freq = torch.pow(
            self.freq_scale, -torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(t.device)
        freq *= t
        emb = torch.cat((torch.sin(freq), torch.cos(freq)), dim=-1)
        return emb.to(t.dtype)


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()
        # if the number of channels is different, we need to use a 1x1 convolution to make them the same
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        # x is of shape (batch_size, in_channels, height, width)
        # time_emb is of shape (batch_size, time_emb_dim)
        residual = x
        x = self.groupnorm1(x)
        x = self.conv1(x)
        x = self.act(x)
        time_emb = self.time_emb_proj(self.act(time_emb))

        # time_emb is of shape (batch_size, out_channels)
        # we want to add it to each pixel in the image
        # so we need to expand it to the same shape as x
        # we can do this by repeating time_emb for each pixel in the image
        # so we need to expand it to the same shape as x
        # we can do this by repeating time_emb for each pixel in the image
        x = x + time_emb.unsqueeze(-1).unsqueeze(-1)
        x = self.groupnorm2(x)
        x = self.conv2(x)
        x = x + self.residual_conv(residual)
        return x


class attention_block(nn.Module):
    def __init__(self, d_embed, num_heads):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, d_embed)
        self.num_heads = num_heads
        self.head_dim = d_embed // num_heads
        self.to_qkv = nn.Linear(d_embed, d_embed * 3)

    def forward(self, x):
        residual = x
        input_shape = x.shape
        b, c, h, w = input_shape
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, h * w, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        weights = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        weights = weights.softmax(dim=-1)
        out = weights @ v
        out = out.permute(0, 2, 1, 3).reshape(input_shape)
        return out + residual


class down_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        layers_per_block,
        downsample=True,
        attention=False,
    ):
        super().__init__()
        self.residual_blocks = nn.ModuleList(
            [
                residual_block(
                    in_channels if i == 0 else out_channels, out_channels, time_emb_dim
                )
                for i in range(layers_per_block)
            ]
        )
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.downsample = nn.Identity()
        if attention:
            self.attention_block = attention_block(out_channels, 4)
        else:
            self.attention_block = nn.Identity()

    def forward(self, x, time_emb):

        for block in self.residual_blocks:
            x = block(x, time_emb)
        x = self.downsample(x)
        x = self.attention_block(x)
        return x


class up_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        layers_per_block,
        upsample=True,
        attention=False,
    ):
        super().__init__()
        self.residual_blocks = nn.ModuleList(
            [
                residual_block(
                    in_channels if i == 0 else out_channels, out_channels, time_emb_dim
                )
                for i in range(layers_per_block)
            ]
        )
        if upsample:
            self.upsample = nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        else:
            self.upsample = nn.Identity()
        if attention:
            self.attention_block = attention_block(out_channels, 4)
        else:
            self.attention_block = nn.Identity()

    def forward(self, x, time_emb):
        for block in self.residual_blocks:
            x = block(x, time_emb)
        x = self.upsample(x)
        x = self.attention_block(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        time_emb_dim=None,
        in_channels=3,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        num_layers_per_block=1,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        self.time_embed_dim = (
            time_emb_dim if time_emb_dim is not None else 4 * block_out_channels[0]
        )
        self.time_embedding = TimeEmbedding(self.time_embed_dim)
        self.down_blocks = nn.ModuleList(
            [
                down_block(
                    block_out_channels[0] if i == 0 else block_out_channels[i - 1],
                    block_out_channels[i],
                    time_emb_dim=self.time_embed_dim,
                    attention=True if i == len(block_out_channels) - 2 else False,
                    downsample=True if i < len(block_out_channels) - 1 else False,
                    layers_per_block=num_layers_per_block,
                )
                for i in range(len(block_out_channels))
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                up_block(
                    in_channels=block_out_channels[i] * 2,
                    out_channels=(
                        block_out_channels[i - 1] if i != 0 else block_out_channels[0]
                    ),
                    time_emb_dim=self.time_embed_dim,
                    attention=True if i == len(block_out_channels) - 2 else False,
                    upsample=True if i < len(block_out_channels) - 1 else False,
                    layers_per_block=num_layers_per_block,
                )
                for i in range(len(block_out_channels))[::-1]
            ]
        )
        self.bottleneck_channels = block_out_channels[-1]
        self.bottleneck = nn.ModuleList(
            [
                residual_block(
                    self.bottleneck_channels,
                    self.bottleneck_channels,
                    self.time_embed_dim,
                ),
                attention_block(self.bottleneck_channels, 4),
            ]
        )
        self.conv_out = nn.Conv2d(
            block_out_channels[0], in_channels, kernel_size=3, padding=(1, 1)
        )

    def forward(self, x, time):
        time_emb = self.time_embedding(time)

        x = self.conv_in(x)
        skip_connections = []
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x, time_emb)
            skip_connections.append(x)
        for layer in self.bottleneck:
            if isinstance(layer, residual_block):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        for i, up_block in enumerate(self.up_blocks):
            residual = skip_connections.pop()
            x = torch.cat((x, residual), dim=1)
            x = up_block(x, time_emb)

        x = self.conv_out(x)
        return x
