from typing import List, Union

import numpy as np
import torch
from torch import Tensor


from core.integrate.utils import *


def is_voxel_in_pixel_grids(
    cam_pose: Tensor,
    cam_intr: Tensor,
    voxel_hash: Tensor,
    voxel_size: float,
    voxel_origin: Union[List[float], np.ndarray, Tensor],  # of shape (3,)
    py_list: List[int],
    px_list: List[int],
):
    """
        given voxel positions, camera poses, and a set of rectangular range of pixels [px_min, px_max), [py_min, py_max)

        Through this we can use device and concour to solve the correspondence problem accordingly
    Inputs:
        hash of shape N
        py of shape Gy+1,
        px of shape Gx+1,
    Returns:
        boolean array of shape N, Gy, Gx, denoting this voxel is in this range or not
    """

    assert cam_intr.size() in [(3, 3), (4, 4)], "[!] `cam_intr' should be of shape (3, 3)."
    assert cam_pose.size() == (4, 4), "[!] `cam_pose' should be of shape (4, 4)."

    sqrt3 = 1.7320508075688772  # maximum radius is voxel_size * sqrt3 / 2
    device = voxel_hash.device

    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    world_c = inhomo2homo(discrete2world(discrete_coord=hash2discrete(voxel_hash), voxel_size=voxel_size, voxel_origin=voxel_origin))

    cam_c = torch.matmul(torch.inverse(cam_pose), world_c.transpose(1, 0)).transpose(1, 0).float()

    z = (cam_c[:, 2] / cam_c[:, 3]).reshape(-1, 1, 1)

    px = (cam_c[:, 0] * fx / cam_c[:, 2]).reshape(-1, 1, 1) + cx - 0.5  # N, 1, 1
    py = (cam_c[:, 1] * fy / cam_c[:, 2]).reshape(-1, 1, 1) + cy - 0.5

    rx = voxel_size * fx / cam_c[:, 2].reshape(-1, 1, 1) * sqrt3 / 2  # maximum radius of this voxel in x and y direction
    ry = voxel_size * fy / cam_c[:, 2].reshape(-1, 1, 1) * sqrt3 / 2  # # N, 1, 1

    gy_min = torch.tensor(py_list[:-1], device=device).reshape(1, -1, 1)
    gy_max = torch.tensor(py_list[+1:], device=device).reshape(1, -1, 1)

    gx_min = torch.tensor(px_list[:-1], device=device).reshape(1, 1, -1)
    gx_max = torch.tensor(px_list[+1:], device=device).reshape(1, 1, -1)

    in_mask = (py + ry >= gy_min) * (py - ry <= gy_max) * (px + rx >= gx_min) * (px - rx <= gx_max) * (z > 0)  # this rx ry is the radius of the oval. This is for the most extreme case that the voxel center is not in the range but its edge is.
    # the last one is to ensure the voxel is in the front

    return in_mask


def voxel_ray_intersect(
    cam_pose: Tensor,
    cam_intr: Tensor,
    voxel_hash: Tensor,
    voxel_size: float,
    voxel_origin: Union[List[float], np.ndarray, Tensor],
    py_min: int,
    py_max: int,
    px_min: int,
    px_max: int,
):  # of shape (3,)
    """
    I don't know about DDA for now, I think it needs iteration, so this is not reasonable. However, this can be done by a series of determinant.
    A ray is O + t D, and a voxel is [x, x+1][y, y+1], [z, z+1], then a ray hit the voxel means the inequality of solution.
    """

    # array dimension rules: NxHxWx{2x3,4}

    device = voxel_hash.device
    H, W = py_max - py_min, px_max - px_min

    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    # first, prepare for the origin
    origin = (cam_pose[:3, 3] / cam_pose[3, 3]).reshape(1, 1, 1, -1, 1)

    # then prepare for ray direction (ax, ay, 1, 0)
    py, px = torch.meshgrid(
        [
            torch.arange(py_min, py_max, device=device),
            torch.arange(px_min, px_max, device=device),
        ],
        indexing="ij",
    )  # now both of shape
    py = py.reshape(1, H, W, 1, 1)
    px = px.reshape(1, H, W, 1, 1)

    d0 = (px - cx + 0.5) / fx
    d1 = (py - cy + 0.5) / fy
    d2 = torch.ones_like(px)
    d3 = torch.zeros_like(px)

    direction = torch.einsum("ij,nhwjx->nhwix", cam_pose, torch.cat([d0, d1, d2, d3], dim=3))[:, :, :, :3, :]  # 1,H,W,3,1, no need to normalizee, the factor is always zero

    # now we have to work on the hash side, the boundarys for hash is its center +/- 0.5*voxel_size
    dr = torch.tensor([-0.5, 0.5], device=device).reshape(1, 1, 1, 1, -1) * voxel_size

    world_c = discrete2world(hash2discrete(voxel_hash), voxel_size=voxel_size, voxel_origin=voxel_origin).reshape(-1, 1, 1, 3, 1)

    determinant = (world_c + dr - origin) / direction

    r = determinant.max(dim=-1)[0].min(dim=-1)[0]
    l = determinant.min(dim=-1)[0].max(dim=-1)[0]  # NxHxW

    # (r > 0 allows a voxel to be just at the origin), (l>0 removes this case)
    is_intersect = (l < r) * (l > 0)  # NxHxW

    # now we calculate the distance and get the closest voxel
    has_interset_voxel = is_intersect.any(dim=0)

    z = l * is_intersect + (~is_intersect) * 1e8  # 1e* means infinite

    indices = has_interset_voxel * (z.argmin(dim=0)) + (~has_interset_voxel) * (-1)

    return indices, z.min(dim=0)[0]


def build_depth_from_voxel(
    cam_pose: Tensor,
    cam_intr: Tensor,
    voxel_hash: Tensor,
    voxel_size: float,
    voxel_origin: Union[List[float], np.ndarray, Tensor],
    H: int,
    W: int,
    pixel_batch_size: int,
    # voxel_batch_size: int,
):
    num = int(np.sqrt(H * W / pixel_batch_size).clip(1))
    py_list = np.linspace(start=0, stop=H, num=num + 1, dtype=int)
    px_list = np.linspace(start=0, stop=W, num=num + 1, dtype=int)

    in_mask = is_voxel_in_pixel_grids(
        cam_pose=cam_pose,
        cam_intr=cam_intr,
        voxel_hash=voxel_hash,
        voxel_size=voxel_size,
        voxel_origin=voxel_origin,
        py_list=py_list,
        px_list=px_list,
    )

    depth = torch.ones(size=(H, W), device=voxel_hash.device) * torch.inf

    for iy in range(num):
        for ix in range(num):
            filtered_hash = voxel_hash[in_mask[:, iy, ix]]

            depth_patch = voxel_ray_intersect(
                cam_pose=cam_pose,
                cam_intr=cam_intr,
                voxel_hash=filtered_hash,
                voxel_size=voxel_size,
                voxel_origin=voxel_origin,
                py_min=py_list[iy],
                py_max=py_list[iy + 1],
                px_min=px_list[ix],
                px_max=px_list[ix + 1],
            )[1]

            depth[py_list[iy] : py_list[iy + 1], px_list[ix] : px_list[ix + 1]] = depth_patch

    return depth


def pixel_dense_voxel_projection(
    cam_pose: Tensor,
    cam_intr: Tensor,
    voxel_hash: Tensor,
    voxel_size: float,
    voxel_origin: Union[List[float], np.ndarray, Tensor],
    H: int,
    W: int,
    pixel_batch_size: int,
):
    num = int(np.sqrt(H * W / pixel_batch_size).clip(1))
    py_list = np.linspace(start=0, stop=H, num=num + 1, dtype=int)
    px_list = np.linspace(start=0, stop=W, num=num + 1, dtype=int)

    in_mask = is_voxel_in_pixel_grids(
        cam_pose=cam_pose,
        cam_intr=cam_intr,
        voxel_hash=voxel_hash,
        voxel_size=voxel_size,
        voxel_origin=voxel_origin,
        py_list=py_list,
        px_list=px_list,
    )

    voxel_corres = torch.zeros(size=(H, W), device=voxel_hash.device, dtype=voxel_hash.dtype)

    for iy in range(num):
        for ix in range(num):
            filtered_hash = voxel_hash[in_mask[:, iy, ix]]

            indices, depth_patch = voxel_ray_intersect(
                cam_pose=cam_pose,
                cam_intr=cam_intr,
                voxel_hash=filtered_hash,
                voxel_size=voxel_size,
                voxel_origin=voxel_origin,
                py_min=py_list[iy],
                py_max=py_list[iy + 1],
                px_min=px_list[ix],
                px_max=px_list[ix + 1],
            )

            voxel_corres[py_list[iy] : py_list[iy + 1], px_list[ix] : px_list[ix + 1]] = filtered_hash[indices] * (~torch.isinf(depth_patch))

    return voxel_hash


