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


def floor_3d(
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
    discrete_coord = torch.floor(discrete_coord / voxel_size).long()
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


def cuda_mem_clear(threshold=95000000000):
    # my gpu have 11GB, so threshold is 10GB
    if torch.cuda.is_available() and torch.cuda.memory_reserved() > threshold:
        # print('empty cache')
        torch.cuda.empty_cache()


def unique_with_indexing(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index


def depth_to_voxel_layer(
    depth: Tensor,
    cam_intr: Tensor,
    cam_pose: Tensor,
    voxel_origin: Tensor,
    voxel_size: float,
    margin: float,
    device: torch.device,
    depth_max: float = 15.0,
):
    """Given a depth image, return the thin layer of voxels that include the depth surface, in the world coordinates
    Args:
        depth_max: maximum depth in meter, used to determin the grid density
    """
    assert cam_intr.size() in [(3, 3), (4, 4)], "[!] `cam_intr' should be of shape (3, 3)."
    assert cam_pose.size() == (4, 4), "[!] `cam_pose' should be of shape (4, 4)."

    H, W = depth.shape[0], depth.shape[1]
    max_factor = 1.7320508075688772

    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    # we want a cubic array of points that can densely fill the true grid world.
    # we want a grid of cubics with a=voxel_size/sqrt(3),
    # nx * a / depth_max * fx = 1, nx = 1 / fx * depth_max / a
    a = voxel_size / max_factor
    nx = torch.ceil(depth_max / (fx * a * 2)).long()
    ny = torch.ceil(depth_max / (fy * a * 2)).long()
    nz = torch.ceil(margin / a + 0.0 * fx).long()

    # making sure the original pixel (0, 0, 0), is in this list
    axis_x = (torch.arange(2 * nx + 1, device=device) - nx) * a
    axis_y = (torch.arange(2 * ny + 1, device=device) - ny) * a
    axis_z = (torch.arange(2 * nz + 1, device=device) - nz) * a

    grid_x, grid_y, grid_z = torch.meshgrid(axis_x, axis_y, axis_z, indexing="ij")
    grid_x, grid_y, grid_z = grid_x.reshape(1, -1), grid_y.reshape(1, -1), grid_z.reshape(1, -1)

    # now we want to apply this cubic array to every pixel
    px, py = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="ij")
    depth = depth.reshape(-1)
    center_x = (px.reshape(-1) - cx + 0.5) / fx * depth
    center_y = (py.reshape(-1) - cy + 0.5) / fy * depth

    # Filter out depth = 0
    filter = depth > 0
    if 0 in filter.size():
        return None
    layer_x = (center_x[filter].reshape(-1, 1) + grid_x).reshape(-1)
    layer_y = (center_y[filter].reshape(-1, 1) + grid_y).reshape(-1)
    layer_z = (depth[filter].reshape(-1, 1) + grid_z).reshape(-1)

    cam_c = torch.stack([layer_x, layer_y, layer_z, torch.ones_like(layer_z)], dim=1)  # homo genious
    world_c_homo = torch.matmul(cam_pose, cam_c.transpose(1, 0)).transpose(1, 0).float()
    world_c = homo2inhomo(world_c_homo)

    hash_c = discrete2hash(
        discretize_3d(
            world_coord=world_c,
            voxel_origin=voxel_origin,
            voxel_size=voxel_size,
        )
    ).unique(sorted=True)

    discrete_c = (hash2discrete(hash_c).reshape(-1, 1, 3) + torch.stack(torch.meshgrid([torch.arange(-1, 2, device=hash_c.device)] * 3, indexing="ij"), dim=-1).reshape(1, -1, 3)).reshape(-1, 3)

    hash_c = discrete2hash(discrete_c).unique(sorted=True)

    return hash_c


def filter_voxel_from_depth_and_get_tsdf(
    hash_c: Tensor,
    world_c: Tensor,
    cam_pose: Tensor,
    depth: Tensor,
    sdf_trunc: float,
    margin: float,
    fx,
    fy,
    cx,
    cy,
    H,
    W,
):
    # convert world coordinates to camera coordinates
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    pix_z = cam_c[:, 2] / cam_c[:, 3]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx - 0.5).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy - 0.5).long()

    # STEP 1: Filter

    # Filter 1: remove pixel outside the frame
    filter_1 = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H) & (pix_z > 0)
    if 0 in filter_1.size():
        return None, None, None, None
    hash_c_f1 = hash_c[filter_1]
    pix_x_f1, pix_y_f1 = pix_x[filter_1], pix_y[filter_1]
    depth_f1 = depth[pix_y_f1, pix_x_f1]

    # Filter 2: remove pixel outside tsdf
    depth_diff = depth_f1 - pix_z[filter_1]
    filter_2 = (depth_f1 > 0) & (depth_diff >= -margin) & (depth_diff <= margin)
    if 0 in filter_2.size():
        return None, None, None, None
    hash_c_f2 = hash_c_f1[filter_2]
    # pix_x_f2, pix_y_f2 = pix_x_f1[filter_2], pix_y_f1[filter_2]
    depth_diff_f2 = depth_diff[filter_2]
    tsdf_f2 = torch.clamp(depth_diff_f2 / sdf_trunc, max=1)

    return hash_c_f2, tsdf_f2


def filter_voxel_from_depth_and_image_and_get_pixels(
    world_c: Tensor,
    cam_pose: Tensor,
    depth: Tensor,
    margin: float,
    fx_d,
    fy_d,
    cx_d,
    cy_d,
    H_d,
    W_d,
    fx_f,
    fy_f,
    cx_f,
    cy_f,
    H_f,
    W_f,
):
    # ! Note that, when we back project pixel-projected voxel to pixel, some voxel's new pixel may deviate a lot from the original pixel, this is due to large variance of depths in this region. Therefore, we need to set a larger margin than sdf_trunc to prevent filtering out these voxels.
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    indices = torch.arange(world_c.size(0), device=world_c.device)

    pix_z = cam_c[:, 2] / cam_c[:, 3]
    pix_x_d = torch.round((cam_c[:, 0] * fx_d / cam_c[:, 2]) + cx_d - 0.5).long()
    pix_y_d = torch.round((cam_c[:, 1] * fy_d / cam_c[:, 2]) + cy_d - 0.5).long()
    pix_x_f = torch.round((cam_c[:, 0] * fx_f / cam_c[:, 2]) + cx_f - 0.5).long()
    pix_y_f = torch.round((cam_c[:, 1] * fy_f / cam_c[:, 2]) + cy_f - 0.5).long()

    # STEP 1: Filter
    # Filter 1: remove pixel outside the frame
    filter_1 = (pix_x_d >= 0) & (pix_x_d < W_d) & (pix_y_d >= 0) & (pix_y_d < H_d) & (pix_x_f >= 0) & (pix_x_f < W_f) & (pix_y_f >= 0) & (pix_y_f < H_f) & (pix_z > 0)
    if 0 in filter_1.size():
        return None, None, None, None
    indices_f1 = indices[filter_1]
    pix_x_d_f1, pix_y_d_f1 = pix_x_d[filter_1], pix_y_d[filter_1]
    pix_x_f_f1, pix_y_f_f1 = pix_x_f[filter_1], pix_y_f[filter_1]
    depth_f1 = depth[pix_y_d_f1, pix_x_d_f1]

    # Filter 2: remove pixel outside tsdf
    depth_diff = depth_f1 - pix_z[filter_1]
    filter_2 = (depth_f1 > 0) & (depth_diff >= -margin) & (depth_diff <= margin)
    if 0 in filter_2.size():
        return None, None, None, None
    indices_f2 = indices_f1[filter_2]
    pix_x_f_f2, pix_y_f_f2 = pix_x_f_f1[filter_2], pix_y_f_f1[filter_2]

    return indices_f2, pix_x_f_f2, pix_y_f_f2


def instance_id_to_one_hot_mask(instance: torch.Tensor):
    """
    Input:
        instance: a tensor of HxW, which dtype of int
    Return:
        a mask tensor of shape n_mask x H x W, of dtype bool
    """
    return torch.nn.functional.one_hot(instance.unique(sorted=True, return_inverse=True)[1]).movedim(-1, 0) > 0
