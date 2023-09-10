from typing import List, Union

import numpy as np
import torch
from torch import Tensor


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


def instance_id_to_one_hot_mask(instance: torch.Tensor, background_id: int):
    """
    Input:
        instance: a tensor of HxW, which dtype of int
    Return:
        a mask tensor of shape n_mask x H x W, of dtype bool
    """
    masks = torch.nn.functional.one_hot(instance.unique(sorted=True, return_inverse=True)[1]).movedim(-1, 0) > 0

    not_background = torch.arange(masks.shape[0], dtype=int) != background_id

    return masks[not_background]


def intrinsic_rescale(
    intr_ori: Tensor,
    H_ori: int,
    W_ori: int,
    H_new: int,
    W_new: int,
):
    intr_new = intr_ori.clone().detach()
    # scale factor
    sx, sy = W_new / W_ori, H_new / H_ori
    intr_new[0, :] = intr_ori[0, :] * sx
    intr_new[1, :] = intr_ori[1, :] * sy

    return intr_new
