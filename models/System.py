# bica_repro.py
"""
Minimal reproduction of embedding-extraction pipeline described (Encoder+BIAM+Noise+Decoder).
Author: assistant reproduction
Requirements: torch, torchvision (for transforms if needed)
Run example at bottom with `python bica_repro.py`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.ConvBlock import ConvBlock
import numpy as np
from typing import Optional


# ---------------------
# Utilities
# ---------------------
def ste_round(x):
    """Straight-through estimator round: forward uses torch.round, backward passes gradients as identity."""
    return (torch.round(x) - x).detach() + x


class DWConv(nn.Module):
    """Depthwise conv with optional BN and activation, used in paper's DWConv units."""

    def __init__(self, channels, kernel=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel, stride=stride,
                            padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


# ---------------------
# BIAM: Bidirectional-Interactive & Context-Aware Module
# ---------------------
class BIAM(nn.Module):
    """
    Implementation notes:
    - The 'unit' in paper: repeated DWConv downsample blocks (we implement with stride=2)
    - pool() refers to repeated downsampling to produce K,V for MHSA.
    - Use PyTorch MultiheadAttention on flattened spatial tokens for MHSA.
    - Local branch: DWConv -> Sigmoid gating.
    """

    def __init__(self, dim, num_heads=4, downsample_stages=2, dw_unit_depth=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # unit: a sequence of DWConv (we will reduce resolution by factor 2 per unit)
        self.downsample_stages = downsample_stages
        # produce K,V via multiple DWConv downsample stages
        units = []
        in_ch = dim
        for i in range(downsample_stages):
            units.append(nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=2, padding=2, groups=in_ch, bias=False))
            units.append(nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False))
            units.append(nn.BatchNorm2d(in_ch))
            units.append(nn.GELU())
        self.unit = nn.Sequential(*units)

        # projectors for Q/K/V (1x1 conv)
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_k = nn.Conv2d(dim, dim, 1)
        self.proj_v = nn.Conv2d(dim, dim, 1)

        # Multihead attention (we will flatten spatial dims)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # local branch
        self.local_dw = DWConv(dim)
        self.local_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # fusion and FFN
        self.fusion_conv = nn.Conv2d(dim, dim, 1)
        self.norm1 = nn.LayerNorm([dim, 1, 1])  # will adjust in forward
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        # Global path: compute pool(K), pool(V) via unit()
        pool_feats = self.unit(x)  # [B, C, H', W']
        # Q from original x (project)
        q = self.proj_q(x)  # [B, C, H, W]
        k = self.proj_k(pool_feats)  # [B, C, H', W']
        v = self.proj_v(pool_feats)  # [B, C, H', W']

        # Flatten spatial dims for mha: shape -> (B, seq_len, C)
        q_tokens = q.flatten(2).transpose(1, 2)  # [B, H*W, C]
        k_tokens = k.flatten(2).transpose(1, 2)  # [B, H'*W', C]
        v_tokens = v.flatten(2).transpose(1, 2)  # [B, H'*W', C]

        # run multihead attention (query from q, key/value from pooled)
        attn_out, _ = self.mha(q_tokens, k_tokens, v_tokens)  # [B, H*W, C]
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)  # back to [B,C,H,W]
        X_global = attn_out

        # Local path
        local = self.local_dw(x)  # [B,C,H,W]
        local_gate = self.local_gate(local)  # [B,C,H,W] gated weights
        X_local = local * local_gate

        # Bidirectional modulation (element-wise gating)
        g1 = torch.sigmoid(X_global)
        g2 = torch.sigmoid(X_local)
        X_local_mod = X_local * g1
        X_global_mod = X_global * g2

        fused = X_local_mod * X_global_mod  # hadamard product
        fused = self.fusion_conv(fused)

        # FFN and residual (paper uses LN+FFN; we use conv FFN)
        # adapt LayerNorm: expect (B,C,H,W) -> permute
        fused_permute = fused.permute(0, 2, 3, 1)  # B H W C
        fused_norm = F.layer_norm(fused_permute, fused_permute.shape[-1:])  # LN over channels
        fused_norm = fused_norm.permute(0, 3, 1, 2)  # B C H W

        out = fused + self.ffn(fused_norm)
        return out


# ---------------------
# Encoder (embedder)
# ---------------------
class Encoder(nn.Module):
    """
    Simplified encoder:
    - Downsample stack (conv stride2)
    - Multiple BIAM blocks at bottleneck
    - Upsample stack with skip connections
    - Input expects concatenated image + watermark-channels along channel dim
    """

    def __init__(self, in_channels=3, wm_channels=1, base_ch=64, num_biam=3):
        super().__init__()
        self.in_ch = in_channels + wm_channels
        # initial convs / downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(self.in_ch, base_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.GELU()
        )
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.bn_d1 = nn.BatchNorm2d(base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.bn_d2 = nn.BatchNorm2d(base_ch * 4)

        # bottleneck BIAM stacks
        self.bottlenecks = nn.ModuleList([BIAM(base_ch * 4) for _ in range(num_biam)])

        # upsample + skip
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)

        self.recon = nn.Sequential(
            nn.Conv2d(base_ch, in_channels, 3, padding=1),
            nn.Tanh()  # produce residual/modified image in [-1,1]
        )

    def forward(self, cover_img, wm_tensor):
        """
        cover_img: [B,3,H,W], values expected -1..1
        wm_tensor: [B,wm_channels,H,W] (watermark channel aligned to image)
        returns: embedded image [B,3,H,W]
        """
        x = torch.cat([cover_img, wm_tensor], dim=1)  # concat channels
        e1 = self.enc1(x)  # [B,base,H,W]
        d1 = F.gelu(self.bn_d1(self.down1(e1)))  # [B,base*2,H/2,W/2]
        d2 = F.gelu(self.bn_d2(self.down2(d1)))  # [B,base*4,H/4,W/4]

        b = d2
        for blk in self.bottlenecks:
            b = blk(b)

        u1 = F.gelu(self.up1(b))  # [B,base*2,H/2,W/2]
        u1 = u1 + d1  # skip
        u2 = F.gelu(self.up2(u1))  # [B,base,H,W]
        u2 = u2 + e1  # skip

        out = self.recon(u2)  # embedded image residual
        # combine with cover to produce final embedded image; paper likely outputs full image
        embedded = cover_img + out * 0.1  # scale residual (lambda-like), adjust as hyperparam
        embedded = torch.clamp(embedded, -1.0, 1.0)
        return embedded


# ---------------------
# Noise layer (differentiable approximations for attacks)
# ---------------------
class NoiseLayer(nn.Module):
    """
    Compose multiple differentiable noise/attack approximations.
    Include Gaussian blur (via conv), Gaussian noise, dropout (spatial), cropout,
    and a simple JPEG-like per-pixel quantization with STE.
    """

    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        # blur kernel simple gaussian
        kernel = self._gaussian_kernel(5, sigma=1.0).to(device)
        self.register_buffer('blur_kernel', kernel)

    def forward(self, img, mode_prob=None):
        # img: [-1,1] -> convert to [0,1] for some ops
        x = (img + 1.0) / 2.0
        B, C, H, W = x.shape

        # random choice of augmentations - for deterministic run set mode_prob
        # gaussian blur
        if torch.rand(1).item() < 0.5:
            x = self._gaussian_blur(x)

        # gaussian noise
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise

        # spatial dropout / cropout
        if torch.rand(1).item() < 0.5:
            # spatial dropout: randomly drop rectangles
            x = self._random_erase(x, p=0.2)

        # JPEG-like quantization (per-pixel scalar quantization as approximation)
        if torch.rand(1).item() < 0.7:
            x = self._jpeg_approx(x, q=50)

        x = torch.clamp(x, 0.0, 1.0)
        return x * 2.0 - 1.0

    def _gaussian_blur(self, x):
        B, C, H, W = x.shape
        # depthwise conv using blur kernel
        pad = self.blur_kernel.shape[-1] // 2
        out = F.conv2d(x.view(-1, 1, H, W), self.blur_kernel, padding=pad, groups=1)
        out = out.view(B, C, H, W)
        return out

    def _gaussian_kernel(self, kernel_size=5, sigma=1.0):
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
        return kernel

    def _random_erase(self, x, p=0.5):
        if torch.rand(1).item() > p:
            return x
        B, C, H, W = x.shape
        for b in range(B):
            if torch.rand(1).item() < 0.5:
                # random rectangle
                rw = int(W * 0.2)
                rh = int(H * 0.2)
                x0 = np.random.randint(0, W - rw + 1)
                y0 = np.random.randint(0, H - rh + 1)
                x[b, :, y0:y0 + rh, x0:x0 + rw] = 0.5
        return x

    def _jpeg_approx(self, x, q=50):
        """Very simple per-pixel scalar quantize with STE to simulate JPEG quantization effect."""
        # x: [0,1]
        scale = 255.0
        X = x * scale
        Q = max(1.0, (100.0 - q) / 50.0 * 10.0 + 1.0)  # crude mapping q->quant step
        Xq = ste_round(X / Q) * Q
        out = Xq / scale
        return out


# ---------------------
# Decoder (extractor)
# ---------------------
class Decoder(nn.Module):
    """
    Decoder: 7 x Conv3x3 (stride 2) -> Flatten -> 2xFC -> Sigmoid output (bit logits)
    We'll parameterize channels to be flexible.
    """

    def __init__(self, in_channels=3, base_ch=32, out_bits=100):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(7):
            next_ch = base_ch * (2 if i >= 2 else 1)  # ramp channels later
            layers.append(ConvBlock(3 if i == 0 else base_ch, next_ch, 3, stride=2, padding=1))
            base_ch = next_ch
        self.conv_stack = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),  # assume input 128x128 -> after 7 stride2 roughly 1-2 dims; adapt if different
            nn.ReLU(),
            nn.Linear(512, out_bits),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,3,H,W]
        y = self.conv_stack(x)
        print(y.shape)
        out = self.fc(y)
        return out  # [B, out_bits] in (0,1)


# ---------------------
# Simple Discriminator (optional)
# ---------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(5):
            layers.append(nn.Conv2d(ch, base_ch * (2 ** i), 3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = base_ch * (2 ** i)
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(ch, 1)

    def forward(self, x):
        h = self.net(x).view(x.shape[0], -1)
        return torch.sigmoid(self.out(h))


# ---------------------
# BCH placeholder: repetition code (for end-to-end runnable demo)
# Replace with real BCH encoder/decoder for production experiments.
# ---------------------
def bch_encode_placeholder(bits: torch.Tensor, repeat: int = 3):
    # bits: [B, k] values 0/1
    # return: [B, k*repeat]
    B, k = bits.shape
    bits_expand = bits.unsqueeze(-1).repeat(1, 1, repeat)  # [B,k,repeat]
    return bits_expand.reshape(B, k * repeat).float()


def bch_decode_placeholder(logits: torch.Tensor, k_orig: int, repeat: int = 3):
    # logits: [B, k*repeat] floats in [0,1]. We majority-vote per repeat group.
    B, n = logits.shape
    assert n == k_orig * repeat
    logits_groups = logits.view(B, k_orig, repeat)
    votes = (logits_groups > 0.5).float().sum(dim=-1)  # counts
    decoded = (votes > (repeat // 2)).long()
    return decoded  # [B,k_orig]


# ---------------------
# Loss helpers
# ---------------------
def bit_error_rate(pred_bits, true_bits):
    # both are 0/1 tensors
    return (pred_bits != true_bits).float().mean().item()


# ---------------------
# Small end-to-end test
# ---------------------
def demo_run(device='cpu'):
    torch.manual_seed(0)
    device = torch.device(device)
    B = 2
    H = W = 128  # working size for demo
    cover = torch.randn(B, 3, H, W, device=device) * 0.1  # small noise cover
    cover = torch.clamp(cover, -1, 1)

    # random watermark bits
    k_bits = 32
    bits = (torch.rand(B, k_bits, device=device) > 0.5).long()

    # encode bits -> BCH placeholder repeated code
    repeat = 3
    encoded = bch_encode_placeholder(bits.float(), repeat=repeat)  # [B, k*repeat]
    # map bits to spatial tensor: simplest: replicate channel across HxW, or better map to single-channel map via tiling
    wm_ch = 1
    # make a wm tensor with shape [B, wm_ch, H, W] by tiling encoded bits across spatial dims
    # For demo, we tile the whole code to the channel plane by reshaping bits to (B,wm_ch,H,W) - quick hack:
    wm_map = encoded.float().unsqueeze(-1).unsqueeze(-1)  # [B,n,1,1]
    wm_map = wm_map.repeat(1, 1, H, W)  # [B,n,H,W]
    # reduce to single channel by averaging groups to fit wm_ch=1 (or use conv to embed)
    wm_tensor = wm_map.mean(dim=1, keepdim=True)  # [B,1,H,W]

    # instantiate models
    enc = Encoder(in_channels=3, wm_channels=1, base_ch=32, num_biam=2).to(device)
    noise = NoiseLayer(device=device).to(device)
    dec = Decoder(in_channels=3, base_ch=16, out_bits=k_bits * repeat).to(device)
    # optional discriminator
    dis = Discriminator().to(device)

    # forward
    embedded = enc(cover, wm_tensor)  # embed
    noised = noise(embedded)  # simulate attack
    # decode
    logits = dec(noised)  # floats in 0..1
    # bch decode placeholder
    decoded = bch_decode_placeholder(logits, k_orig=k_bits, repeat=repeat)  # [B,k]
    ber = bit_error_rate(decoded, bits)
    print(f"Demo BER: {ber:.4f}")
    return {
        'cover': cover, 'wm_bits': bits, 'embedded': embedded, 'noised': noised, 'decoded_bits': decoded
    }


if __name__ == "__main__":
    model = Discriminator()
    a = torch.randn(16, 3, 128, 128)
    out = model(a)
    print(out.shape)
