import torch

from .location_encoder_only import load_location_encoder_checkpoint


def get_satclip_loc_encoder(ckpt_path, device, dtype=torch.float64):
    return load_location_encoder_checkpoint(ckpt_path, device=device, dtype=dtype)
