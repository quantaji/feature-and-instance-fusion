from typing import List, Union

import numpy as np
import torch
from torch import Tensor
from copy import copy


@torch.no_grad()
def instance_2d_map_3d(
    masks: Tensor,
    voxel_instance: Tensor,
    layer_px: Tensor,
    layer_py: Tensor,
    layer_vol_idx: Tensor,
    num_3d_instance: int,
    threshold: float,
):
    """
    This funciton builds a correspondence between 2D instance id to the global 3D instance id.

        masks: one hot boolean tensor of shape (n_masks, H, W), we assume that the background never appears in this tensor, every mask is an instane mask
        voxel_instance: the 3d global instance id of voxels
    """
    device = voxel_instance.device
    H_m, W_m = masks.shape[-2], masks.shape[-1]

    area_order = torch.argsort(masks.sum(dim=(-1, -2)), descending=True, stable=True)  # sort by descending order
    ordered_masks = masks[area_order]

    masks_2d = ordered_masks[:, layer_py, layer_px]  # n_2d_masks, n_pixels

    # it is not efficient to map globel 3d instance id to its one-hot space, because there are many that is not in this view, so we get the unique value of it first
    instance_3d_unique, instance_3d_compact = torch.unique(
        voxel_instance[layer_vol_idx],
        return_inverse=True,
    )
    masks_3d = (torch.nn.functional.one_hot(instance_3d_compact, instance_3d_unique.shape[0]) > 0).movedim(0, 1).unsqueeze(0)  # 1, n_global_labels, n_pixels

    # we also have to remove global_instance_id = 0, since it means unclassified, it should not take part in the matching
    remove_uncls_f = instance_3d_unique != 0  # it is also possible that 0 does not appear in this frame
    instance_3d_unique = instance_3d_unique[remove_uncls_f]
    masks_3d = masks_3d[:, remove_uncls_f, :]
    # del instance_3d_compact

    # compute IoU
    # this step is very memory demanding...
    # a solution is to use a for loop...
    intersect = torch.cat(
        [(masks_3d * mask_2d.reshape(1, 1, -1)).count_nonzero(dim=-1) for mask_2d in masks_2d],
        dim=0,
    )
    union = torch.cat(
        [(masks_3d + mask_2d.reshape(1, 1, -1)).count_nonzero(dim=-1) for mask_2d in masks_2d],
        dim=0,
    )
    IoU = intersect / union  # (n_2d_masks, n_global_labels)
    # del masks_obs

    new_num_3d_instance = num_3d_instance + 0
    temp_map_2d_3d = [0]  # unset maps to unset, this is used in following step
    for z in range(IoU.shape[0]):
        if IoU.shape[1] == 0:
            # in case for the first frame IoU is always of shape (n, 0)
            z_hat = new_num_3d_instance
            new_num_3d_instance += 1

        elif IoU[z].max() > threshold:
            local_index = IoU[z].argmax().item()
            IoU[:, local_index] = 0.0
            z_hat = instance_3d_unique[local_index]

        else:
            z_hat = new_num_3d_instance
            new_num_3d_instance += 1

        temp_map_2d_3d.append(z_hat)

    # del IoU, instance_3d_unique
    temp_map_2d_3d = torch.tensor(temp_map_2d_3d, dtype=torch.int64, device=device)

    # then get a image of update label
    mapped_img = temp_map_2d_3d[
        torch.cat(
            [
                torch.zeros(size=(1, H_m, W_m), dtype=torch.int64, device=device),
                ordered_masks,
            ],
            dim=0,
        ).argmax(dim=0)
    ]

    _, index = masks.sum(dim=(-1, -2)).sort()
    map_2d_3d = temp_map_2d_3d[1:].gather(0, index.argsort(0))  # the mapping acoording to original ordering

    return mapped_img, map_2d_3d, new_num_3d_instance
