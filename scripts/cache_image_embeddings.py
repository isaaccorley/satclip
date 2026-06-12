"""Precompute frozen MoCo ViT-S/16 pre-head image features for all S2-100K patches.

SatCLIP pretraining uses NO image augmentation (confirmed with the author) and the
backbone is frozen, so each patch maps to a fixed 384-d pre-head feature. We compute
it once (center crop 224, /10000, B10-insert -> 13 band) and save it; training then
loads these vectors (no image I/O, no backbone forward) and applies only the trainable
head + location encoder. Order matches S2Geo's usable list so the seed-0 train/val
split is identical to the image-based runs.
"""
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, "/u/isaaccorley/github/satclip/satclip")
from datamodules.s2geo_dataset import S2Geo
from model import SatCLIP

DATA = "/projects/bgtj/isaaccorley/s2-100k-tg-zstd"
OUT = "/projects/bgtj/isaaccorley/s2_embeddings/vit16l40_prehead.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_center = T.CenterCrop(224)


def noaug(sample):
    img = sample["image"].astype(np.float32) / 10000.0           # uint16 12-band -> float
    b10 = np.zeros((1, *img.shape[1:]), dtype=img.dtype)
    img = np.concatenate([img[:10], b10, img[10:]], axis=0)       # -> 13 band (incl B10)
    return {"image": _center(torch.from_numpy(img)), "point": sample["point"]}


def main():
    m = SatCLIP(
        loss_type="soft_loss", soft_loss_penalty="linear", soft_loss_rho_km=20.0, soft_loss_tau_km=10.0,
        embed_dim=256, image_resolution=224, vision_layers="moco_vit16", vision_width=128,
        vision_patch_size=32, in_channels=13, le_type="sphericalharmonics", pe_type="siren",
        frequency_num=32, max_radius=0.001, min_radius=1e-05, harmonics_calculation="analytic",
        legendre_polys=40, sh_embedding_dims=40, num_hidden_layers=2, capacity=512,
    ).to(DEVICE).eval()
    vis = m.visual

    ds = S2Geo(root=DATA, transform=noaug, mode="both")
    dl = DataLoader(ds, batch_size=512, num_workers=12, shuffle=False)
    print(f"embedding {len(ds)} patches on {DEVICE}...", flush=True)

    Zs, Cs, t0 = [], [], time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            imgs = batch["image"].to(DEVICE, non_blocking=True).float()
            z = vis.forward_head(vis.forward_features(imgs), pre_logits=True)  # [B, 384] frozen pre-head
            Zs.append(z.half().cpu().numpy())                                 # float16 -> ~70MB total
            Cs.append(batch["point"].numpy())
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(dl)}  {time.time()-t0:.0f}s", flush=True)
    Z = np.concatenate(Zs)
    C = np.concatenate(Cs).astype(np.float64)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, Z=Z, coords=C)
    print(f"saved Z={Z.shape} ({Z.dtype}) coords={C.shape} -> {OUT}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
