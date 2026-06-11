"""Verify the GPU-transform plan before implementing it:
  1. uint16 survives the full DataLoader->GPU path (from_numpy, collate, pin, .to(cuda)).
  2. torchvision gaussian_blur on CUDA == CPU (precision sanity).
  3. a per-sample-sigma grouped-conv blur == looping torchvision per-sample
     (so we can preserve T.GaussianBlur's per-sample random sigma, batched, on GPU).
"""
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data._utils.collate import default_collate

dev = "cuda"
print("torch", torch.__version__, "cuda", torch.cuda.is_available())

# --- 1. uint16 end-to-end ---
a = np.random.randint(0, 16000, (12, 256, 256), dtype=np.uint16)
t = torch.from_numpy(a); print("1a from_numpy dtype:", t.dtype)
b = default_collate([torch.from_numpy(a) for _ in range(8)]); print("1b collate:", tuple(b.shape), b.dtype)
bp = b.pin_memory(); print("1c pinned:", bp.is_pinned())
g = bp.to(dev, non_blocking=True); print("1d on gpu:", g.dtype, g.device)
x = g.float().div_(10000.0)
B0 = torch.zeros((x.shape[0], 1, *x.shape[2:]), device=dev)
x = torch.cat([x[:, :10], B0, x[:, 10:]], dim=1)
print("1e cast/scale/B10:", tuple(x.shape), x.dtype, "finite:", torch.isfinite(x).all().item())


def gaussian_kernel1d(ksize, sigma):
    # matches torchvision _get_gaussian_kernel1d
    half = (ksize - 1) / 2
    coords = torch.arange(ksize, device=sigma.device, dtype=torch.float32) - half
    pdf = torch.exp(-0.5 * (coords[None, :] / sigma[:, None]) ** 2)  # [B, ksize]
    return pdf / pdf.sum(dim=1, keepdim=True)


def per_sample_blur(img, ksize, sigmas):
    # img [B,C,H,W], sigmas [B]; separable depthwise conv with a per-sample kernel
    B, C, H, W = img.shape
    k1 = gaussian_kernel1d(ksize, sigmas)            # [B, k]
    pad = ksize // 2
    # horizontal then vertical, grouping batch*channel so each sample uses its kernel
    kx = k1.view(B, 1, 1, ksize).expand(B, C, 1, ksize).reshape(B * C, 1, 1, ksize)
    ky = k1.view(B, 1, ksize, 1).expand(B, C, 1, ksize).reshape(B * C, 1, ksize, 1)
    xr = img.reshape(1, B * C, H, W)
    xr = torch.nn.functional.conv2d(torch.nn.functional.pad(xr, (pad, pad, 0, 0), mode="reflect"), kx, groups=B * C)
    xr = torch.nn.functional.conv2d(torch.nn.functional.pad(xr, (0, 0, pad, pad), mode="reflect"), ky, groups=B * C)
    return xr.reshape(B, C, H, W)


# --- 2. torchvision blur CUDA vs CPU ---
xc = torch.rand(8, 13, 256, 256)
yc = F.gaussian_blur(xc, 3, 1.3)
yg = F.gaussian_blur(xc.to(dev), 3, 1.3).cpu()
print("2 cuda vs cpu blur maxdiff:", float((yc - yg).abs().max()))

# --- 3. grouped-conv per-sample blur vs torchvision per-sample ---
sig = torch.empty(8, device=dev).uniform_(0.1, 2.0)
xg = torch.rand(8, 13, 256, 256, device=dev)
mine = per_sample_blur(xg, 3, sig)
ref = torch.stack([F.gaussian_blur(xg[i], 3, float(sig[i])) for i in range(8)])
print("3 grouped-conv vs torchvision per-sample maxdiff:", float((mine - ref).abs().max()))
print("DONE")
