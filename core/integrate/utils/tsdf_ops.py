from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from .voxel_ops import discrete2hash, discrete2world, discretize_3d, hash2discrete, homo2inhomo, inhomo2homo
from .voxel_ray_cast import voxel_batch_divide_from_pixel_grids


def depth_to_voxel_layer(
    depth: Tensor,
    depth_intr: Tensor,
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
    assert depth_intr.size() in [(3, 3), (4, 4)], "[!] `depth_intr' should be of shape (3, 3)."
    assert cam_pose.size() == (4, 4), "[!] `cam_pose' should be of shape (4, 4)."

    H, W = depth.shape[0], depth.shape[1]
    max_factor = 1.7320508075688772

    fx, fy = depth_intr[0, 0], depth_intr[1, 1]
    cx, cy = depth_intr[0, 2], depth_intr[1, 2]

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
    depth_intr: Tensor,
    sdf_trunc: float,
    margin: float,
):
    # convert world coordinates to camera coordinates
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    H, W = depth.shape[0], depth.shape[1]

    fx, fy = depth_intr[0, 0], depth_intr[1, 1]
    cx, cy = depth_intr[0, 2], depth_intr[1, 2]

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
    depth_intr: Tensor,
    feat_intr: Tensor,
    H_f: int,
    W_f: int,
    margin: float,
):
    # ! Note that, when we back project pixel-projected voxel to pixel, some voxel's new pixel may deviate a lot from the original pixel, this is due to large variance of depths in this region. Therefore, we need to set a larger margin than sdf_trunc to prevent filtering out these voxels.
    world2cam = torch.inverse(cam_pose)
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    indices = torch.arange(world_c.size(0), device=world_c.device)

    H_d, W_d = depth.shape[0], depth.shape[1]

    fx_d, fy_d = depth_intr[0, 0], depth_intr[1, 1]
    cx_d, cy_d = depth_intr[0, 2], depth_intr[1, 2]

    pix_z = cam_c[:, 2] / cam_c[:, 3]
    pix_x_d = torch.round((cam_c[:, 0] * fx_d / cam_c[:, 2]) + cx_d - 0.5).long()
    pix_y_d = torch.round((cam_c[:, 1] * fy_d / cam_c[:, 2]) + cy_d - 0.5).long()

    fx_f, fy_f = feat_intr[0, 0], feat_intr[1, 1]
    cx_f, cy_f = feat_intr[0, 2], feat_intr[1, 2]

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


def rough_filter_voxel_given_view(
    cam_pose: Tensor,
    cam_intr: Tensor,
    voxel_hash: Tensor,
    voxel_size: float,
    voxel_origin: Union[List[float], np.ndarray, Tensor],  # of shape (3,)
    H: int,
    W: int,
):
    """
    given a view pose, get all possible voxels that is in this view. No depths envolved
    """
    sqrt3 = 1.7320508075688772  # maximum radius is voxel_size * sqrt3 / 2

    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    world_c = inhomo2homo(discrete2world(discrete_coord=hash2discrete(voxel_hash), voxel_size=voxel_size, voxel_origin=voxel_origin))

    cam_c = torch.matmul(torch.inverse(cam_pose), world_c.transpose(1, 0)).transpose(1, 0).float()

    z = cam_c[:, 2] / cam_c[:, 3]  # N,
    px = (cam_c[:, 0] * fx / cam_c[:, 2]) + cx - 0.5
    py = (cam_c[:, 1] * fy / cam_c[:, 2]) + cy - 0.5

    rx = voxel_size * fx / cam_c[:, 2] * sqrt3 / 2  # maximum radius of this voxel in x and y direction
    ry = voxel_size * fy / cam_c[:, 2] * sqrt3 / 2  # # N, 1, 1

    in_mask = (py + ry >= 0) * (py - ry <= H) * (px + rx >= 0) * (px - rx <= W) * (z > 0)

    return in_mask


def pixel_voxel_corres_given_depth(
    voxel_hash: Tensor,
    voxel_origin: Union[List[float], np.ndarray, Tensor],
    voxel_size: float,
    cam_intr: Tensor,
    cam_pose: Tensor,
    H: int,
    W: int,
    depth: Tensor,
):
    """
    given hashes of voxel, and camera intr, and the H and W, calculate the map from pixel to voxels. It is possible that the pixel's dpeth is inf, so that it does not have correspondence. It is also possible that the depth map we use are from real sensors, which may results in high variance.

    depth can have different resolution and intrinsic w.r.t pixel

    Returns:
        pixel_x, pixel_y, voxel_indices_in_original_voxel_hash_list
    """

    # ! STEP 1, back-project pixel to 3d point clouds, then hash it to get pixel to voxel coorespondence
    depth_interpolated = torch.nn.functional.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest").squeeze(0).squeeze(0)
    _temp_idx = torch.argwhere(depth_interpolated > 0)
    # these are the valid pixels of this image
    py_layer, px_layer = _temp_idx[:, 0], _temp_idx[:, 1]
    # del _temp_idx

    # step 1.2 get the voxel hash and indices of these pixels, and back project to 2D images
    layer_z = depth_interpolated[py_layer, px_layer]  # for voxel remember to add an amount to avoid rounding error

    fx_m, fy_m = cam_intr[0, 0], cam_intr[1, 1]
    cx_m, cy_m = cam_intr[0, 2], cam_intr[1, 2]

    # del depth_interpolated
    layer_x = (px_layer - cx_m + 0.5) / fx_m * layer_z
    layer_y = (py_layer - cy_m + 0.5) / fy_m * layer_z

    layer_hash_c = discrete2hash(
        discretize_3d(
            homo2inhomo(
                torch.matmul(
                    cam_pose,
                    torch.stack(
                        [
                            layer_x,
                            layer_y,
                            layer_z,
                            torch.ones_like(layer_z),
                        ],
                        dim=1,
                    ).transpose(1, 0),
                )
                .transpose(1, 0)
                .float()
            ),
            voxel_origin=voxel_origin,
            voxel_size=voxel_size,
        )
    )
    # del layer_x, layer_y, layer_z

    # ! STEP 2: get the voxels that best fit this view from the tsdf volume
    # note that we might lose some voxels at edge, where its center does not appear in any of the image.
    valid_filter = torch.isin(
        layer_hash_c,
        voxel_hash[
            rough_filter_voxel_given_view(
                cam_intr=cam_intr,
                cam_pose=cam_pose,
                voxel_hash=voxel_hash,
                voxel_origin=voxel_origin,
                voxel_size=voxel_size,
                H=H,
                W=W,
            )
        ],
    )  # this valid_filter is a filter that t

    layer_indices = torch.cat([voxel_hash, layer_hash_c[valid_filter]], dim=0).unique(sorted=True, return_inverse=True)[1][voxel_hash.shape[0] :]

    return px_layer[valid_filter], py_layer[valid_filter], layer_indices
