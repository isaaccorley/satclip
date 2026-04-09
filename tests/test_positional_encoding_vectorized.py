import math

import numpy as np
import torch

from satclip.positional_encoding.common import _cal_freq_list
from satclip.positional_encoding.grid_and_sphere import GridAndSphere
from satclip.positional_encoding.theory import Theory


def _legacy_grid_and_sphere_reference(coords, frequency_num, max_radius, min_radius, name):
    coords_np = np.asarray(coords.cpu())
    batch_size = coords_np.shape[0]
    num_context_pt = 1
    coords_np = coords_np[:, None, :]
    freq_list = _cal_freq_list("geometric", frequency_num, max_radius, min_radius)
    freq_mat = np.repeat(np.expand_dims(freq_list, axis=1), 2, axis=1)
    coords_mat = np.expand_dims(coords_np, axis=3)
    coords_mat = np.expand_dims(coords_mat, axis=4)
    coords_mat = np.repeat(coords_mat, frequency_num, axis=3)
    coords_mat = np.repeat(coords_mat, 2, axis=4)
    spr_embeds = coords_mat * freq_mat

    if name == "grid":
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])
    elif name == "spherec":
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)
        spr_embeds = np.concatenate([lat_sin, lat_cos * lon_cos, lat_cos * lon_sin], axis=-1)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))
    elif name == "spherecplus":
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)
        spr_embeds = np.concatenate(
            [lat_sin, lat_cos, lon_sin, lon_cos, lat_cos * lon_cos, lat_cos * lon_sin],
            axis=-1,
        )
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))
    elif name == "spherem":
        lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
        lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)
        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)
        lat_single_cos = np.cos(lat_single)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)
        spr_embeds = np.concatenate(
            [
                lat_sin,
                lat_cos * lon_single_cos,
                lat_single_cos * lon_cos,
                lat_cos * lon_single_sin,
                lat_single_cos * lon_sin,
            ],
            axis=-1,
        )
    elif name == "spheremplus":
        lon_single = np.expand_dims(coords_mat[:, :, 0, :, :], axis=2)
        lat_single = np.expand_dims(coords_mat[:, :, 1, :, :], axis=2)
        lon_single_sin = np.sin(lon_single)
        lon_single_cos = np.cos(lon_single)
        lat_single_cos = np.cos(lat_single)
        lon = np.expand_dims(spr_embeds[:, :, 0, :, :], axis=2)
        lat = np.expand_dims(spr_embeds[:, :, 1, :, :], axis=2)
        lon_sin = np.sin(lon)
        lon_cos = np.cos(lon)
        lat_sin = np.sin(lat)
        lat_cos = np.cos(lat)
        spr_embeds = np.concatenate(
            [
                lat_sin,
                lat_cos,
                lon_sin,
                lon_cos,
                lat_cos * lon_single_cos,
                lat_single_cos * lon_cos,
                lat_cos * lon_single_sin,
                lat_single_cos * lon_sin,
            ],
            axis=-1,
        )

    return torch.from_numpy(spr_embeds.reshape(coords.shape[0], -1)).to(coords.dtype)


def _legacy_theory_reference(coords, frequency_num, max_radius, min_radius):
    coords_np = np.asarray(coords.cpu())
    batch_size = coords_np.shape[0]
    num_context_pt = coords_np.shape[1]
    freq_list = _cal_freq_list("geometric", frequency_num, max_radius, min_radius)
    freq_mat = np.repeat(np.expand_dims(freq_list, axis=1), 6, axis=1)
    unit_vec1 = np.asarray([1.0, 0.0])
    unit_vec2 = np.asarray([-1.0 / 2.0, math.sqrt(3) / 2.0])
    unit_vec3 = np.asarray([-1.0 / 2.0, -math.sqrt(3) / 2.0])
    angle_mat1 = np.expand_dims(np.matmul(coords_np, unit_vec1), axis=-1)
    angle_mat2 = np.expand_dims(np.matmul(coords_np, unit_vec2), axis=-1)
    angle_mat3 = np.expand_dims(np.matmul(coords_np, unit_vec3), axis=-1)
    angle_mat = np.concatenate([angle_mat1, angle_mat1, angle_mat2, angle_mat2, angle_mat3, angle_mat3], axis=-1)
    angle_mat = np.expand_dims(angle_mat, axis=-2)
    angle_mat = np.repeat(angle_mat, frequency_num, axis=-2)
    angle_mat = angle_mat * freq_mat
    spr_embeds = np.reshape(angle_mat, (batch_size, num_context_pt, -1))
    spr_embeds[:, :, 0::2] = np.sin(spr_embeds[:, :, 0::2])
    spr_embeds[:, :, 1::2] = np.cos(spr_embeds[:, :, 1::2])
    return torch.from_numpy(spr_embeds.reshape(coords.shape[0], -1)).to(coords.dtype)


def test_grid_and_sphere_matches_legacy_numpy_reference():
    coords = torch.tensor(
        [
            [-87.6298, 41.8781],
            [151.2093, -33.8688],
            [2.3522, 48.8566],
            [179.999, 89.999],
        ],
        dtype=torch.float64,
    )
    for name in ["grid", "spherec", "spherecplus", "spherem", "spheremplus"]:
        module = GridAndSphere(frequency_num=8, max_radius=0.01, min_radius=0.00001, name=name)
        actual = module(coords)
        expected = _legacy_grid_and_sphere_reference(coords, 8, 0.01, 0.00001, name)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_theory_matches_legacy_numpy_reference():
    coords = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[-5.0, 6.0], [7.0, -8.0]],
        ],
        dtype=torch.float64,
    )
    module = Theory(frequency_num=6, max_radius=10000, min_radius=1000)
    actual = module(coords)
    expected = _legacy_theory_reference(coords, 6, 10000, 1000)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
