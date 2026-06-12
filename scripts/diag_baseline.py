"""Diagnose why base_spatial (satclip_loss + spatial) stalls while soft_loss runs fine.

Builds the EXACT pipeline (spatial batching + gpu transform) and times a manual
train loop for satclip_loss vs soft_loss on the same GPU. If satclip_loss is
much slower per-batch, it's the loss path; if both are equal, the loss is
exonerated (=> node / Lightning / I/O). Run on a GPU node.
"""
import os, sys, time
import torch
sys.path.insert(0, "/u/isaaccorley/github/satclip/satclip")
from datamodules.s2geo_dataset import S2GeoDataModule
from main import SatCLIPLightningModule

DATA = "/projects/bgtj/isaaccorley/s2-100k-tg-zstd"
COMMON = dict(embed_dim=256, image_resolution=256, vision_layers=4, vision_width=128,
    vision_patch_size=32, in_channels=13, le_type="sphericalharmonics", legendre_polys=32,
    sh_embedding_dims=32, frequency_num=16, max_radius=0.01, min_radius=0.00001,
    capacity=256, num_hidden_layers=2)


def run(loss_type, penalty="linear", n_batches=25):
    dm = S2GeoDataModule(data_dir=DATA, batch_size=512, num_workers=8, transform="gpu",
                         pin_memory=True, spatial_batching=True)
    dm.setup("fit")
    dl = dm.train_dataloader()
    lm = SatCLIPLightningModule(loss_type=loss_type, soft_loss_penalty=penalty,
        soft_loss_rho_km=20.0, soft_loss_tau_km=10.0, **COMMON).cuda()
    opt = torch.optim.AdamW(lm.parameters(), lr=1e-4)
    it = iter(dl)
    t_data = t_step = 0.0
    print(f"\n[{loss_type}] timing {n_batches} batches...", flush=True)
    torch.cuda.synchronize()
    for i in range(n_batches):
        t0 = time.time()
        batch = next(it)
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        batch = lm.on_after_batch_transfer(batch, 0)
        torch.cuda.synchronize(); t_data += time.time() - t0
        t1 = time.time()
        opt.zero_grad()
        loss = lm.common_step(batch, i)
        loss.backward(); opt.step()
        torch.cuda.synchronize(); t_step += time.time() - t1
        if i in (0, 4, n_batches - 1):
            print(f"  batch {i}: data+xform={ (time.time()-t0)*1000:.0f}ms loss={float(loss):.3f}", flush=True)
    n = n_batches
    print(f"[{loss_type}] avg data+xform={t_data/n*1000:.0f}ms/batch  compute={t_step/n*1000:.0f}ms/batch"
          f"  total={(t_data+t_step)/n*1000:.0f}ms/batch  => ~{(t_data+t_step)/n*162:.1f}s/epoch", flush=True)


if __name__ == "__main__":
    print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(0))
    run("satclip_loss")
    run("soft_loss")
    print("\nDIAG DONE")
