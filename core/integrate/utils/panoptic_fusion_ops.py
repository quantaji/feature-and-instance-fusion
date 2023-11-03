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
        sorted=True,
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


@torch.no_grad()
def guided_instance_2d_map_3d(
    masks: Tensor,
    guided_voxel_label: Tensor,
    layer_px: Tensor,
    layer_py: Tensor,
    layer_vol_idx: Tensor,
    Io2D_threshold: float = 0.5,
):
    """
    This funciton builds a correspondence between 2D instance id to the global 3D instance id. The guided_voxel_label is the "guidance" and is assumed to be ground truth. The guided_voxel_label is assumed to have minimal label of 1, instead of 0, to avoid interference with un classified id=0

        masks: one hot boolean tensor of shape (n_masks, H, W), we assume that the background never appears in this tensor, every mask is an instane mask
        voxel_instance: the 3d global instance id of voxels
    """
    device = guided_voxel_label.device
    H_m, W_m = masks.shape[-2], masks.shape[-1]

    # preprocessing
    # if masks have intersection, then for each pixel choose the one with smallest area
    masks_area = masks.count_nonzero(dim=[1, 2])
    temp = torch.cat(
        [
            (masks_area.max() + 1)
            * torch.ones(
                size=[1] + list(masks.shape[1:]),
                device=masks.device,
                dtype=masks_area.dtype,
            ),
            (masks_area.reshape(-1, 1, 1) * masks + (masks_area.max() + 1) * (~masks)),
        ],
        dim=0,
    )
    instance_id_2d = temp.argmin(dim=0)  # (H, W), 0 denotes no class, id > 0 means the original position
    masks_unique = torch.nn.functional.one_hot(instance_id_2d, num_classes=temp.shape[0]).movedim(-1, 0)[1:]

    masks_2d = masks_unique[:, layer_py, layer_px]  # n_2d_masks, n_pixels

    # it is not efficient to map globel 3d instance id to its one-hot space, because there are many that is not in this view, so we get the unique value of it first
    instance_3d_unique, instance_3d_compact = torch.unique(
        guided_voxel_label[layer_vol_idx],
        return_inverse=True,
        sorted=True,
    )
    masks_3d = (torch.nn.functional.one_hot(instance_3d_compact, instance_3d_unique.shape[0]) > 0).movedim(0, 1).unsqueeze(0)  # 1, n_global_labels, n_pixels

    # compute IoU
    # this step is very memory demanding...
    # a solution is to use a for loop...
    intersect = torch.cat(
        [(masks_3d * mask_2d.reshape(1, 1, -1)).count_nonzero(dim=-1) for mask_2d in masks_2d],
        dim=0,
    )
    # union = torch.cat(
    #     [(masks_3d + mask_2d.reshape(1, 1, -1)).count_nonzero(dim=-1) for mask_2d in masks_2d],
    #     dim=0,
    # )

    # # ! Matching Step 1: use 2D mask with highest IoU (>50% to guarantee uniqueness)
    # IoU = intersect / union  # (n_2d_masks, n_global_labels)
    # max_IoU, argmax_IoU = IoU.max(dim=0)  # n_global_labels

    # # ! Matching Step 2: for all 2D masks with high Io2DArea
    masks_2d_area = masks_2d.count_nonzero(dim=-1)  # n_2d_masks,
    Io2D = intersect / masks_2d_area.reshape(-1, 1)  # (n_2d_masks, n_global_labels)
    max_Io2D, argmax_Io2D = Io2D.max(dim=1)  # n_2d_masks

    map_2d_3d = instance_3d_unique[argmax_Io2D * (max_Io2D > Io2D_threshold)]

    temp_map_2d_3d = torch.tensor([0] + map_2d_3d.tolist(), dtype=torch.int64, device=device)
    # add zero at the beginning, so that un assigned pixels also have their position

    # then get a image of update label
    mapped_img = temp_map_2d_3d[instance_id_2d]

    return mapped_img, map_2d_3d
