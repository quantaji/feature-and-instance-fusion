from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as f
from fast_pytorch_kmeans import KMeans
from PIL import Image
from pycocotools import mask as mask_utils


def boolean_masks_to_unint8_masks(boolean_masks: np.ndarray):
    """
    for a binary masks of shape (n, H, W), convert it into shape (ceil(n/8), H, W), to save mem
    """
    return np.packbits(boolean_masks, axis=0, bitorder="big")


def uint8_masks_to_boolean_masks(uint8_masks: np.ndarray, n_masks: int):
    """
    additional n_mask should be provided, otherwise, it expands to 8*ceil(n/8)
    """
    return np.unpackbits(uint8_masks, axis=0, bitorder="big")[:n_masks]


def isolate_boolean_masks(boolean_masks: np.ndarray):
    """
    Some times, the masks may intersect, we want the most isolated, fractalrized masks. The returned mask is integered.
    for a binary masks of shape (n, H, W). The return is a interger mask of shape (H, W)
    """
    return np.unique(
        np.moveaxis(boolean_masks, 0, -1).copy().view(f"a{boolean_masks.shape[0]}"),
        return_inverse=True,
    )[
        1
    ].reshape(boolean_masks.shape[1:3])


@torch.no_grad()
def boolean_mask_to_random_feats(
    boolean_masks: Union[torch.Tensor, np.ndarray, List[torch.Tensor], List[np.ndarray]],
    feat_dim: int,
    device: str,
) -> torch.Tensor:
    """
    convert a boolean mask of shape (n, H, W), to a dense feature, each pixels in the same mask share a random uniform vector draw from gaussian distribution. If masks coinsides, then it is the linear composition of variace random feature
    """
    H, W = boolean_masks[0].shape  # any of the above type support this

    dense_feats = torch.randn((H, W, feat_dim), device=device, dtype=torch.float32) * 1e-16

    direction_feats = f.normalize(torch.randn((len(boolean_masks), feat_dim), device=device, dtype=torch.float32), dim=-1)

    for i, mask in enumerate(boolean_masks):
        if isinstance(mask, torch.Tensor):
            seg_idx = torch.argwhere(mask)
        else:
            seg_idx = torch.argwhere(torch.from_numpy(mask))  # this resolves device issue
        dense_feats[seg_idx[:, 0], seg_idx[:, 1]] += direction_feats[i, :]

    return dense_feats


def resize_feats(dense_feats: torch.Tensor, H: int, W: int):
    """
    feats is assumed to have shape (H, W, feat_dim)
    """
    dense_feats = dense_feats.movedim(2, 0).unsqueeze(0)
    dense_feats = f.interpolate(dense_feats, size=[H, W], mode="nearest")
    dense_feats = dense_feats.squeeze(0).movedim(0, 2)

    return dense_feats


def resize_masks(masks: torch.Tensor, H: int, W: int):
    """
    masks is assumed to have shape (n_mask, H, W)
    """

    return f.interpolate(masks.to(torch.uint8).unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.9])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def get_kmeans_labels(K: int, feats: torch.Tensor):
    """
    feats: (H, W, n)
    """
    kmeans = KMeans(n_clusters=K, mode="euclidean")
    labels = kmeans.fit_predict(feats.reshape(-1, feats.shape[-1]).cuda()).reshape(feats.shape[0], feats.shape[1])

    return labels.detach().cpu().numpy()
