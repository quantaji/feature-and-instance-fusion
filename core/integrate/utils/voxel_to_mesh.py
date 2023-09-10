from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from .voxel_ops import discrete2hash, discretize_3d, hash2discrete, homo2inhomo


DIRECTIONS = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
]  # 6, 3

CUBE_VERTS = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
]  # relative to (-0.5, -0.5, -0.5), 8,3

CUBE_FACES = [
    [[7, 5, 6], [5, 4, 6]],
    [[2, 0, 3], [0, 1, 3]],
    [[3, 7, 2], [7, 6, 2]],
    [[5, 1, 4], [1, 0, 4]],
    [[3, 1, 7], [1, 5, 7]],
    [[6, 4, 2], [4, 0, 2]],
]  # of shape 6,2,3


def get_mesh_from_voxel(voxel_hash: Tensor):
    """This funtion gets the meshed of cubes of voxel, not marching cube algorithm. The return verts are shifted by (+0.5, +0.5, +0.5), remember to remove it by - 0.5 * voxel_size in real work coordinates"""
    discrete_c = hash2discrete(voxel_hash)  # N, 3

    neighbor_c = discrete_c.reshape(-1, 1, 3) + torch.tensor(DIRECTIONS).to(discrete_c).reshape(1, 6, 3)  # N, 6, 3
    neighbor_h = discrete2hash(neighbor_c)
    add_this_face = ~torch.isin(neighbor_h, voxel_hash).reshape(-1)  # N*6

    voxel_verts = (torch.tensor(CUBE_VERTS).to(voxel_hash).reshape(1, 8, 3) + discrete_c.reshape(-1, 1, 3)).reshape(-1, 3)  # N*8 ,3
    voxel_faces = torch.arange(discrete_c.shape[0], device=voxel_hash.device, dtype=voxel_hash.dtype).reshape(-1, 1, 1, 1) * 8 + torch.tensor(CUBE_FACES).to(voxel_hash).reshape(1, 6, 2, 3)  # N, 6, 2, 3
    voxel_faces = voxel_faces.reshape(-1, 2, 3)  # N*6, 2, 3

    selected_verts_idx, selected_faces = voxel_faces[add_this_face].unique(sorted=True, return_inverse=True)
    selected_verts = voxel_verts[selected_verts_idx]

    return selected_verts, selected_faces.reshape(-1, 3)
