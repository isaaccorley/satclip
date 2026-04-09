import torch
from torch import nn
from einops import rearrange, repeat

from .common import _cal_freq_list

"""
Grid, SphereC, SphereCPlus, SphereM, SphereMPlus location encoders
"""


FEATURE_COUNTS = {
    "grid": 4,
    "spherec": 6,
    "spherecplus": 12,
    "spherem": 10,
    "spheremplus": 16,
}


class GridAndSphere(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, coord_dim=2, frequency_num=16,
                 max_radius=0.01, min_radius=0.00001,
                 freq_init="geometric", name="grid"):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridAndSphere, self).__init__()

        if name not in FEATURE_COUNTS:
            raise ValueError(f"unsupported grid/sphere encoder: {name}")

        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.name = name
        self.embedding_dim = self.cal_embedding_dim()

    def cal_embedding_dim(self):
        return FEATURE_COUNTS[self.name] * self.frequency_num

    def cal_freq_list(self):
        freq_list = _cal_freq_list(
            self.freq_init,
            self.frequency_num,
            self.max_radius,
            self.min_radius,
        )
        self.register_buffer(
            "freq_list",
            torch.as_tensor(freq_list, dtype=torch.float64),
            persistent=False,
        )

    def _pack_features(self, *features):
        return rearrange(
            repeat(torch.stack(features, dim=-1), "b p f feat -> b p f feat dup", dup=2),
            "b p f feat dup -> b p (f feat dup)",
        )

    def forward(self, coords):
        dtype = coords.dtype
        freq_list = self.freq_list.to(device=coords.device, dtype=dtype)

        # add 1 context point dimension (unused here)
        coords = rearrange(coords, "n c -> n 1 c")
        coords_base = rearrange(coords, "b p c -> b p c 1") * rearrange(freq_list, "f -> 1 1 1 f")
        lon = coords_base[:, :, 0, :]
        lat = coords_base[:, :, 1, :]
        lon_sin = torch.sin(lon)
        lon_cos = torch.cos(lon)
        lat_sin = torch.sin(lat)
        lat_cos = torch.cos(lat)

        if self.name == "grid":
            spr_embeds = rearrange(
                torch.stack(
                    (torch.sin(coords_base), torch.cos(coords_base)),
                    dim=-1,
                ),
                "b p c f s -> b p (c f s)",
            )
        elif self.name == "spherec":
            spr_embeds = self._pack_features(
                lat_sin,
                lat_cos * lon_cos,
                lat_cos * lon_sin,
            )
        elif self.name == "spherecplus":
            spr_embeds = self._pack_features(
                lat_sin,
                lat_cos,
                lon_sin,
                lon_cos,
                lat_cos * lon_cos,
                lat_cos * lon_sin,
            )
        elif self.name == "spherem":
            lon_single_sin = torch.sin(rearrange(coords[:, :, 0], "b p -> b p 1"))
            lon_single_cos = torch.cos(rearrange(coords[:, :, 0], "b p -> b p 1"))
            lat_single_cos = torch.cos(rearrange(coords[:, :, 1], "b p -> b p 1"))
            spr_embeds = self._pack_features(
                lat_sin,
                lat_cos * lon_single_cos,
                lon_cos * lat_single_cos,
                lat_cos * lon_single_sin,
                lon_sin * lat_single_cos,
            )
        elif self.name == "spheremplus":
            lon_single_sin = torch.sin(rearrange(coords[:, :, 0], "b p -> b p 1"))
            lon_single_cos = torch.cos(rearrange(coords[:, :, 0], "b p -> b p 1"))
            lat_single_cos = torch.cos(rearrange(coords[:, :, 1], "b p -> b p 1"))
            spr_embeds = self._pack_features(
                lat_sin,
                lat_cos,
                lon_sin,
                lon_cos,
                lat_cos * lon_single_cos,
                lon_cos * lat_single_cos,
                lat_cos * lon_single_sin,
                lon_sin * lat_single_cos,
            )

        return rearrange(spr_embeds, "b p d -> b (p d)").to(dtype)
