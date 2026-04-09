import torch
from torch import nn
import math
from einops import rearrange

from .common import _cal_freq_list

"""
Theory based location encoder
"""
class Theory(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, coord_dim=2, frequency_num=16,
                 max_radius=10000, min_radius=1000, freq_init="geometric"):
        """
        Args:
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(Theory, self).__init__()
        self.frequency_num = frequency_num
        self.coord_dim = coord_dim
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.freq_init = freq_init

        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()

        # there unit vectors which is 120 degree apart from each other
        self.register_buffer(
            "unit_vectors",
            torch.tensor(
                [
                    [1.0, 0.0],
                    [-1.0 / 2.0, math.sqrt(3) / 2.0],
                    [-1.0 / 2.0, -math.sqrt(3) / 2.0],
                ],
                dtype=torch.float64,
            ),
            persistent=False,
        )

        self.embedding_dim = self.cal_embedding_dim()

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

    def cal_embedding_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(2 * 3 * self.frequency_num)

    def forward(self, coords):
        dtype = coords.dtype
        freq_list = self.freq_list.to(device=coords.device, dtype=dtype)
        unit_vectors = self.unit_vectors.to(device=coords.device, dtype=dtype)

        angles = coords @ unit_vectors.T
        phases = rearrange(angles, "b p dir -> b p 1 dir") * rearrange(freq_list, "f -> 1 1 f 1")
        spr_embeds = torch.stack((torch.sin(phases), torch.cos(phases)), dim=-1)

        return rearrange(spr_embeds, "b p f dir trig -> b (p f dir trig)").to(dtype)
