# -*- coding: utf-8 -*-
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from typing import Tuple, Optional, Sequence, Union, List
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from einops import rearrange ,repeat
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, DropPath
import numpy as np
import math
import time
import os
#import matplotlib.pyplot as plt

import pywt
from torch.autograd import Function

# -----------------------------
# constants and small helpers
# -----------------------------
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def _bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'b h w c -> b c h w')

def _bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'b c h w -> b h w c')

# ============================================================
# =============== 1D Wavelet (channel axis) ==================
# 正交(haar/db/sym/coif) —— Polyphase + Periodization (PR)
# 双正交(bior2.2/bior4.4) —— Lifting (PR)
# 以及：可学习的通道 lifting（LCL, CDF 9/7 模板）
# ============================================================

def _wname(n: str) -> str:
    n = n.lower()
    alias = {
        "cdf97": "bior4.4", "9/7": "bior4.4", "cdf9/7": "bior4.4",
        "cdf53": "bior2.2", "5/3": "bior2.2",
        "db1": "haar",
    }
    return alias.get(n, n)

def _is_orthogonal_wavelet(w: pywt.Wavelet) -> bool:
    return bool(getattr(w, "orthogonal", False))

def _roll(x, s):
    return torch.roll(x, shifts=s, dims=-1)

def _split_even_odd_lenL(x: torch.Tensor):
    idx_e = torch.arange(0, x.size(-1), 2, device=x.device)
    idx_o = torch.arange(1, x.size(-1), 2, device=x.device)
    return x.index_select(-1, idx_e), x.index_select(-1, idx_o)

# ------- lifting constants -------
_C97 = dict(alpha=-1.586134342059924,
            beta =-0.052980118572961,
            gamma= 0.882911075530934,
            delta= 0.443506852043971,
            K=1.149604398)
_C53 = dict(alpha=-0.5, beta=0.25, K=1.0)

# ------- periodic circular 1D correlation for per-channel coeffs -------
def _circ_corr1d(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, L], k: [C,1,K] (groups=C). circular pad left K-1, conv1d as correlation.
    """
    K = k.shape[-1]
    if K > 1:
        x_pad = F.pad(x, (K-1, 0), mode="circular")
    else:
        x_pad = x
    return F.conv1d(x_pad, k, stride=1, padding=0, groups=x.shape[1])

def _mk_kernel_per_channel(C: int, coeffs: torch.Tensor, device, dtype) -> torch.Tensor:
    k = coeffs.to(device=device, dtype=dtype).view(1,1,-1).repeat(C,1,1)
    return k

# ------- orthogonal polyphase (analysis/synthesis) -------
def _polyphase_filters_ortho(w: pywt.Wavelet, device, dtype):
    dec_lo = torch.tensor(w.dec_lo, dtype=dtype, device=device)
    dec_hi = torch.tensor(w.dec_hi, dtype=dtype, device=device)
    rec_lo = torch.tensor(w.rec_lo, dtype=dtype, device=device)
    rec_hi = torch.tensor(w.rec_hi, dtype=dtype, device=device)
    h0e, h0o = dec_lo[::2], dec_lo[1::2]
    h1e, h1o = dec_hi[::2], dec_hi[1::2]
    g0e, g0o = rec_lo[::2], rec_lo[1::2]
    g1e, g1o = rec_hi[::2], rec_hi[1::2]
    return (h0e, h0o, h1e, h1o), (g0e, g0o, g1e, g1o)

def _dwt1d_ortho_periodic(x: torch.Tensor, w: pywt.Wavelet) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [B, C=1, L=C_channels]  (我们在通道维做长度 L=C 的 1D 变换)
    xe, xo = _split_even_odd_lenL(x)  # [B,1,L/2] each
    (h0e,h0o,h1e,h1o), _ = _polyphase_filters_ortho(w, x.device, x.dtype)
    cA = _circ_corr1d(xe, _mk_kernel_per_channel(1, h0e, x.device, x.dtype)) + \
         _circ_corr1d(xo, _mk_kernel_per_channel(1, h0o, x.device, x.dtype))
    cD = _circ_corr1d(xe, _mk_kernel_per_channel(1, h1e, x.device, x.dtype)) + \
         _circ_corr1d(xo, _mk_kernel_per_channel(1, h1o, x.device, x.dtype))
    return cA, cD

def _iwt1d_ortho_periodic(cA: torch.Tensor, cD: torch.Tensor, w: pywt.Wavelet) -> torch.Tensor:
    _, (g0e,g0o,g1e,g1o) = _polyphase_filters_ortho(w, cA.device, cA.dtype)
    xe = _circ_corr1d(cA, _mk_kernel_per_channel(1, g0e, cA.device, cA.dtype)) + \
         _circ_corr1d(cD, _mk_kernel_per_channel(1, g1e, cA.device, cA.dtype))
    xo = _circ_corr1d(cA, _mk_kernel_per_channel(1, g0o, cA.device, cA.dtype)) + \
         _circ_corr1d(cD, _mk_kernel_per_channel(1, g1o, cA.device, cA.dtype))
    B, C1, Le = xe.shape
    out = xe.new_empty(B, C1, 2*Le)
    out[..., 0::2] = xe
    out[..., 1::2] = xo
    return out

# ------- lifting (biorthogonal CDF) -------
def _lifting97_fwd(x):
    s, d = _split_even_odd_lenL(x)
    s_l, s_r = _roll(s, 1), _roll(s, -1); d = d + _C97['alpha'] * (s_l + s_r)
    d_l, d_r = _roll(d, 1), _roll(d, -1); s = s + _C97['beta']  * (d_l + d_r)
    s_l, s_r = _roll(s, 1), _roll(s, -1); d = d + _C97['gamma'] * (s_l + s_r)
    d_l, d_r = _roll(d, 1), _roll(d, -1); s = s + _C97['delta'] * (d_l + d_r)
    s = s * _C97['K']; d = d / _C97['K']
    return s, d

def _lifting97_inv(s, d):
    s = s / _C97['K']; d = d * _C97['K']
    d_l, d_r = _roll(d, 1), _roll(d, -1); s = s - _C97['delta'] * (d_l + d_r)
    s_l, s_r = _roll(s, 1), _roll(s, -1); d = d - _C97['gamma'] * (s_l + s_r)
    d_l, d_r = _roll(d, 1), _roll(d, -1); s = s - _C97['beta']  * (d_l + d_r)
    s_l, s_r = _roll(s, 1), _roll(s, -1); d = d - _C97['alpha'] * (s_l + s_r)
    out = s.new_empty(*s.shape[:-1], s.size(-1)+d.size(-1))
    out[..., 0::2] = s; out[..., 1::2] = d
    return out

def _lifting53_fwd(x):
    s, d = _split_even_odd_lenL(x)
    s_l, s_r = _roll(s, 1), _roll(s, -1); d = d + _C53['alpha'] * (s_l + s_r)
    d_l, d_r = _roll(d, 1), _roll(d, -1); s = s + _C53['beta']  * (d_l + d_r)
    return s, d

def _lifting53_inv(s, d):
    d_l, d_r = _roll(d, 1), _roll(d, -1); s = s - _C53['beta']  * (d_l + d_r)
    s_l, s_r = _roll(s, 1), _roll(s, -1); d = d - _C53['alpha'] * (s_l + s_r)
    out = s.new_empty(*s.shape[:-1], s.size(-1)+d.size(-1))
    out[..., 0::2] = s; out[..., 1::2] = d
    return out

# ------- ChannelDWT / ChannelIDWT (generic, fixed filters) -------
class ChannelDWT(nn.Module):
    """
    在“通道维”(C)上做 1D 小波分解：x->[low, high]，默认 bior4.4。
    说明：通道维没有天然邻域语义，但工程上按长度=C 的序列做 DWT 是可逆的。
    """
    def __init__(self, wavelet: str = 'bior4.4'):
        super().__init__()
        self.wavelet_name = _wname(wavelet)
        self.w = pywt.Wavelet(self.wavelet_name)
        self.is_ortho = _is_orthogonal_wavelet(self.w)
        if (not self.is_ortho) and self.wavelet_name not in ("bior4.4", "bior2.2"):
            raise NotImplementedError(f"ChannelDWT暂只支持正交(haar/db/sym/coif)与 bior2.2/bior4.4，收到: {wavelet}")

    def forward(self, x: Tensor):
        # x: [B, C, H, W]，需要 C 为偶数
        B, C, H, W = x.shape
        assert C % 2 == 0, f"ChannelDWT 需要通道为偶数，当前 C={C}"
        t = x.permute(0,2,3,1).reshape(B*H*W, 1, C)  # [BHW,1,C]
        if self.is_ortho:
            s, d = _dwt1d_ortho_periodic(t, self.w)   # [BHW,1,C/2]
        else:
            if self.wavelet_name == "bior4.4":
                s, d = _lifting97_fwd(t)
            else:
                s, d = _lifting53_fwd(t)
        low  = s.reshape(B, H, W, C//2).permute(0,3,1,2).contiguous()
        high = d.reshape(B, H, W, C//2).permute(0,3,1,2).contiguous()
        return low, high

class ChannelIDWT(nn.Module):
    """
    通道维 IDWT：接回 (low, high) -> [B,C,H,W]。
    """
    def __init__(self, wavelet: str = 'bior4.4'):
        super().__init__()
        self.wavelet_name = _wname(wavelet)
        self.w = pywt.Wavelet(self.wavelet_name)
        self.is_ortho = _is_orthogonal_wavelet(self.w)
        if (not self.is_ortho) and self.wavelet_name not in ("bior4.4", "bior2.2"):
            raise NotImplementedError(f"ChannelIDWT暂只支持正交(haar/db/sym/coif)与 bior2.2/bior4.4，收到: {wavelet}")

    def forward(self, low: Tensor, high: Tensor):
        B, C2, H, W = low.shape
        assert high.shape == (B, C2, H, W)
        s = low.permute(0,2,3,1).reshape(B*H*W, 1, C2)
        d = high.permute(0,2,3,1).reshape(B*H*W, 1, C2)
        if self.is_ortho:
            x = _iwt1d_ortho_periodic(s, d, self.w)   # [BHW,1,2*C2]
        else:
            if self.wavelet_name == "bior4.4":
                x = _lifting97_inv(s, d)
            else:
                x = _lifting53_inv(s, d)
        x = x.reshape(B, H, W, 2*C2).permute(0,3,1,2).contiguous()
        return x

# ------- Learnable Channel Lifting (LCL, CDF 9/7 模板，可学习，PR) -------
class LearnableChannelDWT97(nn.Module):
    """
    在“通道维(C)”上做 1D lifting 9/7（CDF 9/7 模板，但系数可学习）。
    任意参数组合+对应逆算子 => 严格PR。参数初始化为 CDF 9/7 的经典值。
    """
    def __init__(self, init_from_cdf97: bool = True):
        super().__init__()
        if init_from_cdf97:
            a, b, g, d, K = (_C97['alpha'], _C97['beta'], _C97['gamma'], _C97['delta'], _C97['K'])
        else:
            a=b=g=d=0.0; K=1.0
        self.alpha = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(g, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(d, dtype=torch.float32))
        self.logK  = nn.Parameter(torch.tensor(math.log(K), dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W], C 必须为偶数
        返回 (low, high)，形状 [B, C/2, H, W]
        """
        B, C, H, W = x.shape
        assert C % 2 == 0, f"LearnableChannelDWT97 需要偶数通道，当前 C={C}"
        t = x.permute(0,2,3,1).reshape(B*H*W, 1, C)  # [BHW,1,C]
        s, d = _split_even_odd_lenL(t)
        a = self.alpha; b = self.beta; g = self.gamma; dlt = self.delta
        K  = torch.exp(self.logK)
        s_l, s_r = _roll(s, 1), _roll(s, -1); d = d + a * (s_l + s_r)
        d_l, d_r = _roll(d, 1), _roll(d, -1); s = s + b * (d_l + d_r)
        s_l, s_r = _roll(s, 1), _roll(s, -1); d = d + g * (s_l + s_r)
        d_l, d_r = _roll(d, 1), _roll(d, -1); s = s + dlt* (d_l + d_r)
        s = s * K
        d = d / K
        low  = s.reshape(B, H, W, C//2).permute(0,3,1,2).contiguous()
        high = d.reshape(B, H, W, C//2).permute(0,3,1,2).contiguous()
        return low, high

class LearnableChannelIDWT97(nn.Module):
    """ 与上面 DWT 配对的精确逆变换（使用相同可学习参数） """
    def __init__(self, dwt: LearnableChannelDWT97):
        super().__init__()
        self.dwt = dwt  # 共享同一组参数

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        B, C2, H, W = low.shape
        s = low.permute(0,2,3,1).reshape(B*H*W, 1, C2)
        d = high.permute(0,2,3,1).reshape(B*H*W, 1, C2)
        a = self.dwt.alpha; b = self.dwt.beta; g = self.dwt.gamma; dlt = self.dwt.delta
        K  = torch.exp(self.dwt.logK)
        s = s / K
        d = d * K
        d_l, d_r = _roll(d, 1), _roll(d, -1); s = s - dlt * (d_l + d_r)
        s_l, s_r = _roll(s, 1), _roll(s, -1); d = d - g   * (s_l + s_r)
        d_l, d_r = _roll(d, 1), _roll(d, -1); s = s - b   * (d_l + d_r)
        s_l, s_r = _roll(s, 1), _roll(s, -1); d = d - a   * (s_l + s_r)
        x = s.new_empty(*s.shape[:-1], s.size(-1) + d.size(-1))
        x[..., 0::2] = s
        x[..., 1::2] = d
        x = x.reshape(B, H, W, 2*C2).permute(0,3,1,2).contiguous()
        return x

# ============================================================
# =============== 2D DWT/IDWT (space) 与你一致 ===============
# ============================================================

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)
            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters.to(dtype=torch.float32))

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32))

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# ---------------------------
# conv helpers
# ---------------------------
def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

def find_named_module(module, query):
    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    return next((b for n, b in module.named_buffers() if n == query), None)

def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')
        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)
    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')
        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))
    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    if not module: return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')
    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )

# ---------------------------
# residual blocks
# ---------------------------
class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid_ch = min(in_ch, out_ch) // 2
        self.conv1 = conv1x1(in_ch, mid_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.skip = conv1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = x
        out = self.conv1(out); out = self.relu1(out)
        out = self.conv2(out); out = self.relu2(out)
        out = self.conv3(out)
        return out + identity

class ResidualBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv =  conv(in_ch, out_ch, kernel_size=5, stride=2)
        self.res1 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res2 = ResidualBottleneckBlock(out_ch, out_ch)
        self.res3 = ResidualBottleneckBlock(out_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return out

class ResidualBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch, in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch, in_ch)
        self.conv = deconv(in_ch, out_ch, kernel_size=5, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.conv(out)
        return out

# ---------------------------
# WMSA（加入通道小波包裹，默认 bior4.4）
# ---------------------------
class WMSA(nn.Module):
    """Window/Shifted-Window MSA (Swin-style) + 可选通道小波包裹"""
    def __init__(self, input_dim, output_dim, head_dim, window_size, type,
                 channel_wavelet: str = 'bior4.4'):  # 默认 bior4.4
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        assert input_dim % head_dim == 0, "input_dim 必须能被 head_dim 整除"
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        assert type in ['W', 'SW']
        self.type = type

        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        rel_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_params = nn.Parameter(torch.zeros(rel_size, self.n_heads))
        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads)
            .permute(2, 0, 1).contiguous()
        )

        # —— 通道 DWT 包裹 —— #
        self.use_channel_wavelet = channel_wavelet is not None
        if self.use_channel_wavelet:
            self._cdwt = ChannelDWT(wavelet=channel_wavelet)
            self._cidwt = ChannelIDWT(wavelet=channel_wavelet)

        self.register_buffer('_rel_index', self._build_relation_index(window_size), persistent=False)

    @staticmethod
    def _build_relation_index(p: int) -> torch.Tensor:
        i = torch.arange(p); j = torch.arange(p)
        ii, jj = torch.meshgrid(i, j, indexing='ij')
        cord = torch.stack([ii.reshape(-1), jj.reshape(-1)], dim=1)
        relation = cord[:, None, :] - cord[None, :, :] + p - 1
        return relation.long()

    def relative_embedding(self) -> torch.Tensor:
        p = self.window_size
        rel = self._rel_index.to(self.relative_position_params.device)
        rp = self.relative_position_params  # [H, 2p-1, 2p-1]
        return rp[:, rel[..., 0], rel[..., 1]]  # [H, p*p, p*p]

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def _pre_wavelet(self, x_bhwc: torch.Tensor) -> torch.Tensor:
        x_bchw = _bhwc_to_bchw(x_bhwc)
        C = x_bchw.size(1)
        assert C % 2 == 0, "通道数必须为偶数以进行通道 DWT"
        low, high = self._cdwt(x_bchw)
        x_w = torch.cat([low, high], dim=1)
        return _bchw_to_bhwc(x_w)

    def _post_wavelet(self, y_bhwc: torch.Tensor) -> torch.Tensor:
        y_bchw = _bhwc_to_bchw(y_bhwc)
        C = y_bchw.size(1); assert C % 2 == 0
        low, high = torch.split(y_bchw, C // 2, dim=1)
        y_rec = self._cidwt(low, high)
        return _bchw_to_bhwc(y_rec)

    def forward(self, x):
        """
        x: [B, H, W, C]
        """
        if self.use_channel_wavelet:
            x = self._pre_wavelet(x)

        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))

        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c',
                      p1=self.window_size, p2=self.window_size)
        h_windows, w_windows = x.size(1), x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c')

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')

        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = F.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c',
                           w1=h_windows, p1=self.window_size)

        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2), dims=(1, 2))

        if self.use_channel_wavelet:
            output = self._post_wavelet(output)
        return output

# ---------------------------
# DWConv + GLU
# ---------------------------
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = hidden_features//2
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x)) * v
        x = self.fc2(x)
        return x

# ---------------------------
# Scale + Block + Swin wrapper
# ---------------------------
class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)
    def forward(self, x):
        return x * self.scale

class ResScaleConvolutionGateBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)  # 默认开启通道小波（bior4.4）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = ConvolutionalGLU(input_dim, input_dim * 4)
        self.res_scale_1 = Scale(input_dim, init_value=1.0)
        self.res_scale_2 = Scale(input_dim, init_value=1.0)

    def forward(self, x):
        x = self.res_scale_1(x) + self.drop_path(self.msa(self.ln1(x)))
        x = self.res_scale_2(x) + self.drop_path(self.mlp(self.ln2(x)))
        return x

class SwinBlockWithConvMulti(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, block=ResScaleConvolutionGateBlock, block_num=2, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.block_num = block_num
        for i in range(block_num):
            ty = 'W' if i%2==0 else 'SW'
            self.layers.append(block(input_dim, input_dim, head_dim, window_size, drop_path, type=ty))
        self.conv = conv(input_dim, output_dim, 3, 1)
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col+1, padding_row, padding_row+1))
        trans_x = Rearrange('b c h w -> b h w c')(x)
        for i in range(self.block_num):
            trans_x = self.layers[i](trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        trans_x = self.conv(trans_x)
        if resize:
            x = F.pad(x, (-padding_col, -padding_col-1, -padding_row, -padding_row-1))
        return trans_x + x

# ---------------------------
# Spatial attention / dense / MSAggregation
# ---------------------------
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvWithDW(nn.Module):
    def __init__(self, input_dim=320, output_dim=320):
        super(ConvWithDW, self).__init__()
        self.in_trans = nn.Conv2d(input_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, stride=1, groups=output_dim, bias=True)
        self.act2 = nn.GELU()
        self.out_trans = nn.Conv2d(output_dim, output_dim, kernel_size=1, padding=0, stride=1, bias=True)
    def forward(self, x):
        x = self.in_trans(x); x = self.act1(x)
        x = self.dw_conv(x);  x = self.act2(x)
        x = self.out_trans(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, dim=320):
        super(DenseBlock, self).__init__()
        self.layer_num = 3
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.GELU(), ConvWithDW(dim, dim)) for _ in range(self.layer_num)
        ])
        self.proj = nn.Conv2d(dim*(self.layer_num+1), dim, kernel_size=1, padding=0, stride=1, bias=True)
    def forward(self, x):
        outputs = [x]
        for i in range(self.layer_num):
            outputs.append(self.conv_layers[i](outputs[-1]))
        x = self.proj(torch.cat(outputs, dim=1))
        return x

class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super(MultiScaleAggregation, self).__init__()
        self.s = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.spatial_atte = SpatialAttentionModule()
        self.dense = DenseBlock(dim)
    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        s = self.s(x)
        s_out = self.dense(s)
        x = s_out * self.spatial_atte(s_out)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

# ---------------------------
# 字典跨注意力（加入通道小波包裹，默认 bior4.4）
# ---------------------------
class MutiScaleDictionaryCrossAttentionGLU(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_rate=4, head_num=20, qkv_bias=True,
                 channel_wavelet: str = 'bior4.4'):
        super().__init__()

        dict_dim = 32 * head_num
        self.head_num = head_num

        self.scale = nn.Parameter(torch.ones(head_num, 1, 1))
        self.x_trans = nn.Linear(input_dim, dict_dim, bias=qkv_bias)

        self.ln_scale = nn.LayerNorm(dict_dim)
        self.msa = MultiScaleAggregation(dict_dim)

        self.lnx = nn.LayerNorm(dict_dim)
        self.q_trans = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.dict_ln = nn.LayerNorm(dict_dim)
        self.k = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)

        self.linear = nn.Linear(dict_dim, dict_dim, bias=qkv_bias)
        self.ln_mlp = nn.LayerNorm(dict_dim)

        self.mlp = ConvolutionalGLU(dict_dim, mlp_rate * dict_dim)
        self.output_trans = nn.Sequential(nn.Linear(dict_dim, output_dim))
        self.softmax = torch.nn.Softmax(dim=-1)

        self.res_scale_1 = Scale(dict_dim, init_value=1.0)
        self.res_scale_2 = Scale(dict_dim, init_value=1.0)
        self.res_scale_3 = Scale(dict_dim, init_value=1.0)

        # —— 通道 DWT 包裹 —— #
        self.use_channel_wavelet = channel_wavelet is not None
        if self.use_channel_wavelet:
            self._cdwt = ChannelDWT(wavelet=channel_wavelet)
            self._cidwt = ChannelIDWT(wavelet=channel_wavelet)

    def _pre_wavelet(self, x_bhwc: torch.Tensor) -> torch.Tensor:
        x_bchw = _bhwc_to_bchw(x_bhwc)
        C = x_bchw.size(1); assert C % 2 == 0
        low, high = self._cdwt(x_bchw)
        x_w = torch.cat([low, high], dim=1)
        return _bchw_to_bhwc(x_w)

    def _post_wavelet(self, y_bhwc: torch.Tensor) -> torch.Tensor:
        y_bchw = _bhwc_to_bchw(y_bhwc)
        C = y_bchw.size(1); assert C % 2 == 0
        low, high = torch.split(y_bchw, C // 2, dim=1)
        y_rec = self._cidwt(low, high)
        return _bchw_to_bhwc(y_rec)

    def forward(self, x, dt):
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b h w c')   # BHWC
        x = self.x_trans(x)                      # -> dict_dim

        x = self.msa(self.ln_scale(x)) + self.res_scale_1(x)

        if self.use_channel_wavelet:
            x = self._pre_wavelet(x)

        shortcut = x
        x = self.lnx(x)
        x = self.q_trans(x)
        x = rearrange(x, 'b h w c -> b c h w')

        q = rearrange(x, 'b (e c) h w -> b e (h w) c', e=self.head_num)
        dt = self.dict_ln(dt)
        k = self.k(dt)
        k = rearrange(k, 'b n (e c) -> b e n c', e=self.head_num)
        dt = rearrange(dt, 'b n (e c) -> b e n c', e=self.head_num)

        self.scale = self.scale.to(q.device)
        sim = torch.einsum('benc,bedc->bend', q, k)
        sim = sim * self.scale
        probs = self.softmax(sim)
        output = torch.einsum('bend,bedc->benc', probs, dt)
        output = rearrange(output, 'b e (h w) c -> b h w (e c) ', h=H, w=W)

        output = self.linear(output) + self.res_scale_2(shortcut)
        output = self.mlp(self.ln_mlp(output)) + self.res_scale_3(output)

        if self.use_channel_wavelet:
            output = self._post_wavelet(output)

        output = self.output_trans(output)
        output = rearrange(output, 'b h w c -> b c h w')
        return output

# ---------------------------
# CASM: 自适应尺度调制（编码/解码一致）
# ---------------------------
class ScaleModNet(nn.Module):
    """
    Content-Adaptive Scale Modulation:
    输入与 cc_mean/cc_scale 相同的 support，输出与 scale 同形状的调制“偏置” r，
    factor = 1 + alpha * tanh(r) ∈ (1-alpha, 1+alpha)，初始≈1（零偏置）
    """
    def __init__(self, in_channels: int, out_channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_channels, 224, stride=1, kernel_size=3), nn.GELU(),
            conv(224, 128, stride=1, kernel_size=3), nn.GELU(),
            conv(128, out_channels, stride=1, kernel_size=3),
        )
        last = list(self.net.children())[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(self, support: torch.Tensor):
        r = self.net(support)
        factor = 1.0 + torch.tanh(r) * torch.clamp(self.alpha, 0.0, 1.0)
        return factor

# ---------------------------
# 主模型
# ---------------------------
class ChWDTA_spit8_wave_transformer_bior44_learnable(CompressionModel):
    def __init__(
        self,
        head_dim=[8, 16, 32, 32, 16, 8],
        drop_path_rate=0,
        N=192, M=320,
        num_slices=8, max_support_slices=4,
        wavelet='bior4.4',                    # 默认通道小波（固定滤波器）
        use_learnable_channel: bool = True,  # 开启可学习通道 lifting (LCL)
        use_casm: bool = True,                # 开启 CASM 自适应尺度调制
        casm_alpha_init: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.head_dim = head_dim
        self.window_size = 8
        self.max_support_slices = 8
        self.num_slices = 8
        dim = N
        self.M = M
        input_image_channel = 3
        output_image_channel = 3
        feature_dim = [96, 144, 256]
        self.use_casm = use_casm

        basic_block = [ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock, ResScaleConvolutionGateBlock]
        swin_block = [SwinBlockWithConvMulti, SwinBlockWithConvMulti, SwinBlockWithConvMulti]
        block_num = [1, 2, 12]

        dict_num = 128
        dict_head_num = 20
        dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(torch.randn([dict_num, dict_dim]), requires_grad=True)

        prior_dim = M
        mlp_rate=4
        qkv_bias=True
        self.dt_cross_attention = nn.ModuleList(
            MutiScaleDictionaryCrossAttentionGLU(
                input_dim=M*2+(M//self.num_slices)*i,
                output_dim=M,
                head_num=dict_head_num,
                mlp_rate=mlp_rate,
                qkv_bias=qkv_bias
            ) for i in range(num_slices)
        )

        self.m_down1 = [swin_block[0](feature_dim[0], feature_dim[0], self.head_dim[0], self.window_size, 0, basic_block[0], block_num=block_num[0])] + \
                      [ResidualBottleneckBlockWithStride(feature_dim[0], feature_dim[1])]
        self.m_down2 = [swin_block[1](feature_dim[1], feature_dim[1], self.head_dim[1], self.window_size, 0, basic_block[1], block_num=block_num[1])] + \
                      [ResidualBottleneckBlockWithStride(feature_dim[1], feature_dim[2])]
        self.m_down3 = [swin_block[2](feature_dim[2], feature_dim[2], self.head_dim[2], self.window_size, 0, basic_block[2], block_num=block_num[2])] + \
                      [conv(feature_dim[2], M, kernel_size=5, stride=2)]

        self.m_up1 = [swin_block[2](feature_dim[2], feature_dim[2], self.head_dim[3], self.window_size, 0, basic_block[2], block_num=block_num[2])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1])]
        self.m_up2 = [swin_block[1](feature_dim[1], feature_dim[1], self.head_dim[4], self.window_size, 0, basic_block[1], block_num=block_num[1])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0])]
        self.m_up3 = [swin_block[0](feature_dim[0], feature_dim[0], self.head_dim[5], self.window_size, 0, basic_block[0], block_num=block_num[0])] + \
                      [ResidualBottleneckBlockWithUpsample(feature_dim[0], output_image_channel)]

        self.g_a = nn.Sequential(*[ResidualBottleneckBlockWithStride(input_image_channel, feature_dim[0])] + self.m_down1 + self.m_down2 + self.m_down3)
        self.g_s = nn.Sequential(*[deconv(M, feature_dim[2], kernel_size=5, stride=2)] + self.m_up1 + self.m_up2 + self.m_up3)

        self.ha_down = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [conv(N, 192, kernel_size=3, stride=2)]

        self.h_a = nn.Sequential(
            *[ResidualBottleneckBlockWithStride(M, N)] + \
            self.ha_down
        )

        self.hs_up1 = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [ResidualBottleneckBlockWithUpsample(N, M)]

        self.h_z_s1 = nn.Sequential(
            *[deconv(192, N, kernel_size=3, stride=2)] + \
            self.hs_up1
        )

        self.hs_up2 = [SwinBlockWithConvMulti(N, N, 32, 4, 0, ResScaleConvolutionGateBlock, block_num=1)] + \
                      [ResidualBottleneckBlockWithUpsample(N, M)]

        self.h_z_s2 = nn.Sequential(
            *[deconv(192, N, kernel_size=3, stride=2)] + \
            self.hs_up2
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i, 8) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i, 8) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320*2 + (320//self.num_slices)*min(i+1, 9) + prior_dim, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, (320//self.num_slices), stride=1, kernel_size=3),
            ) for i in range(self.num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

        # ===== 通道方向小波（两种模式：固定通用 / 可学习 lifting） =====
        self.use_learnable_channel = use_learnable_channel
        if use_learnable_channel:
            self.ch_dwt1 = LearnableChannelDWT97(init_from_cdf97=True)
            self.ch_idwt1 = LearnableChannelIDWT97(self.ch_dwt1)
            self.ch_dwt2 = LearnableChannelDWT97(init_from_cdf97=True)
            self.ch_idwt2 = LearnableChannelIDWT97(self.ch_dwt2)
            self.ch_dwt3 = LearnableChannelDWT97(init_from_cdf97=True)
            self.ch_idwt3 = LearnableChannelIDWT97(self.ch_dwt3)
        else:
            self.channel_dwt_1 = ChannelDWT(wavelet=wavelet)
            self.channel_idwt_1 = ChannelIDWT(wavelet=wavelet)
            self.channel_dwt_2 = ChannelDWT(wavelet=wavelet)
            self.channel_idwt_2 = ChannelIDWT(wavelet=wavelet)
            self.channel_dwt_3 = ChannelDWT(wavelet=wavelet)
            self.channel_idwt_3 = ChannelIDWT(wavelet=wavelet)

        # ===== CASM：自适应尺度调制 =====
        if self.use_casm:
            self.scale_mod_transforms = nn.ModuleList(
                ScaleModNet(
                    in_channels=320*2 + (320//self.num_slices)*min(i, 8) + M,
                    out_channels=(320//self.num_slices),
                    alpha_init=casm_alpha_init
                ) for i in range(self.num_slices)
            )

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    # --------- helpers for channel wavelet dispatch ---------
    def _cdwt(self, i, x):
        if self.use_learnable_channel:
            if i == 1: return self.ch_dwt1(x)
            if i == 2: return self.ch_dwt2(x)
            if i == 3: return self.ch_dwt3(x)
        else:
            if i == 1: return self.channel_dwt_1(x)
            if i == 2: return self.channel_dwt_2(x)
            if i == 3: return self.channel_dwt_3(x)
        raise RuntimeError

    def _cidwt(self, i, low, high):
        if self.use_learnable_channel:
            if i == 1: return self.ch_idwt1(low, high)
            if i == 2: return self.ch_idwt2(low, high)
            if i == 3: return self.ch_idwt3(low, high)
        else:
            if i == 1: return self.channel_idwt_1(low, high)
            if i == 2: return self.channel_idwt_2(low, high)
            if i == 3: return self.channel_idwt_3(low, high)
        raise RuntimeError

    def forward(self, x):
        b = x.size(0)
        dt = self.dt.repeat([b, 1, 1])
        y = self.g_a(x)

        # 通道方向小波两级
        ch_low, ch_high = self._cdwt(1, y)
        ch_low_low, ch_low_high = self._cdwt(2, ch_low)
        ch_high_low, ch_high_high = self._cdwt(3, ch_high)
        C_channels = ch_high_low.shape[1]
        y_shape = y.shape[2:]
        y_input = torch.cat([ch_low_low, ch_low_high, ch_high_low, ch_high_high], dim=1)

        z = self.h_a(y_input)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        y_slices = y_input.chunk(self.num_slices, 1)
        y_hat_slices, y_likelihood = [], []
        mu_list, scale_list = [], []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            mu_list.append(mu)
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # ===== CASM（可选）=====
            if self.use_casm:
                factor = self.scale_mod_transforms[slice_index](support)
                factor = factor[:, :, :y_shape[0], :y_shape[1]]
                scale = scale * factor
            # =======================

            scale_list.append(scale)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        means = torch.cat(mu_list, dim=1)
        scales = torch.cat(scale_list, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        ch_low_low = y_hat[:, :C_channels, :, :]
        ch_low_high = y_hat[:, C_channels:2*C_channels, :, :]
        ch_high_low = y_hat[:, 2*C_channels:3*C_channels, :, :]
        ch_high_high = y_hat[:, 3*C_channels:4*C_channels, :, :]
        y_ch_low = self._cidwt(2, ch_low_low, ch_low_high)
        y_ch_high = self._cidwt(3, ch_high_low, ch_high_high)
        y_hat_channel = self._cidwt(1, y_ch_low, y_ch_high)

        x_hat = self.g_s(y_hat_channel)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "para":{"means": means, "scales":scales, "y":y}
        }

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)
        #return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        b = x.size(0)
        dt = self.dt.repeat([b, 1, 1])
        y = self.g_a(x)

        ch_low, ch_high = self._cdwt(1, y)
        ch_low_low, ch_low_high = self._cdwt(2, ch_low)
        ch_high_low, ch_high_high = self._cdwt(3, ch_high)
        C_channels = ch_high_low.shape[1]
        y_shape = y.shape[2:]
        y_input = torch.cat([ch_low_low, ch_low_high, ch_high_low, ch_high_high], dim=1)

        z = self.h_a(y_input)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)

        y_slices = y_input.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if self.use_casm:
                factor = self.scale_mod_transforms[slice_index](support)
                factor = factor[:, :, :y_shape[0], :y_shape[1]]
                scale = scale * factor

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs
        scales = torch.max(scales, torch.tensor(0.11, device=inputs.device, dtype=inputs.dtype))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        b = z_hat.size(0)
        dt = self.dt.repeat([b, 1, 1])
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            dict_info = self.dt_cross_attention[slice_index](query, dt)
            support = torch.cat([query] + [dict_info], dim=1)

            mu = self.cc_mean_transforms[slice_index](support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]
            scale = self.cc_scale_transforms[slice_index](support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            if self.use_casm:
                factor = self.scale_mod_transforms[slice_index](support)
                factor = factor[:, :, :y_shape[0], :y_shape[1]]
                scale = scale * factor

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            rv = rv.to(mu.device)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        C_channels =  int((y_hat.shape[1])//4)
        ch_low_low = y_hat[:, :C_channels, :, :]
        ch_low_high = y_hat[:, C_channels:2*C_channels, :, :]
        ch_high_low = y_hat[:, 2*C_channels:3*C_channels, :, :]
        ch_high_high = y_hat[:, 3*C_channels:4*C_channels, :, :]
        y_ch_low = self._cidwt(2, ch_low_low, ch_low_high)
        y_ch_high = self._cidwt(3, ch_high_low, ch_high_high)
        y_hat_channel = self._cidwt(1, y_ch_low, y_ch_high)

        x_hat = self.g_s(y_hat_channel).clamp_(0, 1)
        return {"x_hat": x_hat}

# ---------------------------
# quick self-check / demo
# ---------------------------
def _bits_per_pixel(lik_y: torch.Tensor, lik_z: torch.Tensor, img_hw):
    H, W = img_hw
    Ry = torch.sum(-torch.log2(lik_y + 1e-9), dim=[1,2,3]) / (H * W)
    Rz = torch.sum(-torch.log2(lik_z + 1e-9), dim=[1,2,3]) / (H * W)
    return Ry + Rz, Ry, Rz

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ====== 你可以在这里切换两种模式 ======
    # 1) 固定通道小波（默认 bior4.4），不启用 CASM：
    # model = DCAE_spit8_wave_transformer(wavelet='bior4.4', use_learnable_channel=False, use_casm=False).to(device)

    # 2) 启用 LCL（可学习通道 lifting）+ 启用 CASM（论文两项新点）：
    model = DCAE_spit8_wave_transformer(
        use_learnable_channel=True,
        use_casm=True,
        casm_alpha_init=0.5
    ).to(device)
    # ======================================

    model.update()
    model.eval()

    B, C, H, W = 1, 3, 256, 256
    x = torch.rand(B, C, H, W, device=device)

    with torch.no_grad():
        out = model(x)
        x_hat = out["x_hat"].clamp(0,1)
        lik_y = out["likelihoods"]["y"]
        lik_z = out["likelihoods"]["z"]
        bpp, bpp_y, bpp_z = _bits_per_pixel(lik_y, lik_z, (H, W))
        mse = F.mse_loss(x_hat, x)
        psnr = -10 * torch.log10(mse)

    print(f"[Forward] bpp={bpp.item():.4f}, bpp_y={bpp_y.item():.4f}, bpp_z={bpp_z.item():.4f}, PSNR={psnr.item():.2f}dB")

    # 压缩-解压（当前实现可靠支持 batch=1）
    comp = model.compress(x)
    decomp = model.decompress(comp["strings"], comp["shape"])
    x_rec = decomp["x_hat"]
    print(f"[Codec] round-trip ok, rec shape={tuple(x_rec.shape)}")
