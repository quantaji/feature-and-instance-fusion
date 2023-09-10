from typing import List, Union

import numpy as np
import torch
from torch import Tensor


def discretize_3d(
    world_coord: Tensor,
    voxel_size: float,
    voxel_origin: Union[List[float], np.ndarray, Tensor] = [0, 0, 0],
):
    """Given (N, 3) world coordinates, voxel origin and voxel size, return the (i_x, i_y, i_z) integer coordinates
    Args:
        world_coord (Tensor): world coordinates of shape (N1, N2, ..., 3)
        voxel_origin (Tensor, or numpy.ndarray, or list): the x,y,z origin of this voxel coordinates
        voxel_size (float): voxel size
    """
    assert world_coord.size(-1) == 3, "[!] `world_coord`'s last coordinate is not shape 3."
    original_size = world_coord.size()

    voxel_origin = Tensor(voxel_origin).reshape(-1).to(world_coord.device)
    assert voxel_origin.size() == (3,), "[!] `voxel_origin` is not of shape 3."
    voxel_origin.reshape(1, 3)

    discrete_coord = world_coord.reshape(-1, 3) - voxel_origin
    discrete_coord = torch.round(discrete_coord / voxel_size).long()
    discrete_coord = discrete_coord.reshape(original_size)

    return discrete_coord


def discrete2world(
    discrete_coord: Tensor,
    voxel_size: float,
    voxel_origin: Union[List[float], np.ndarray, Tensor] = [0, 0, 0],
):
    assert discrete_coord.size(-1) == 3, "[!] `world_coord`'s last coordinate is not shape 3."
    original_size = discrete_coord.size()

    voxel_origin = Tensor(voxel_origin).reshape(-1).to(discrete_coord.device)
    assert voxel_origin.size() == (3,), "[!] `voxel_origin` is not of shape 3."
    voxel_origin.reshape(1, 3)

    world_c = discrete_coord.reshape(-1, 3) * voxel_size + voxel_origin

    return world_c.reshape(original_size)


def inhomo2homo(inhomo_coord: Tensor):
    """All vectors assume to be of shape (N, 3) or (N, 4)"""
    return torch.cat([inhomo_coord, torch.ones_like(inhomo_coord[:, :1])], dim=-1)


def homo2inhomo(homo_coord: Tensor):
    return homo_coord[:, :3] / homo_coord[:, 3].reshape(-1, 1)


def discrete2hash(discrete_coord: Tensor):
    """
    Given a tensor of (N1, N2, ..., Ni, 3), return a tensor of shape (N1, N2, ..., Ni), note that each integer value cannot exceed the range of [ -1048576, 1048575]. If we choose a voxel size of 0.001m = 1mm, this can give as about 1km of range.
    """
    assert discrete_coord.size(-1) == 3, "[!] `discrete_coord`'s last coordinate is not shape 3."
    assert discrete_coord.dtype == torch.int64, "[!] `discrete_coord`'s dtype is not int64."
    assert (discrete_coord.max() < (1 << 20)) and (discrete_coord.min() >= (-1 << 20)), "[!] maximum integer value exceeds."

    x = discrete_coord.select(dim=-1, index=0) + (1 << 20)
    y = discrete_coord.select(dim=-1, index=1) + (1 << 20)
    z = discrete_coord.select(dim=-1, index=2) + (1 << 20)

    return x * (1 << 42) + y * (1 << 21) + z


def hash2discrete(hash: Tensor):
    x = torch.div(hash, (1 << 42), rounding_mode="floor") - (1 << 20)
    yz = torch.remainder(hash, (1 << 42))
    y = torch.div(yz, (1 << 21), rounding_mode="floor") - (1 << 20)
    z = torch.remainder(yz, (1 << 21)) - (1 << 20)

    return torch.stack([x, y, z], dim=-1)
