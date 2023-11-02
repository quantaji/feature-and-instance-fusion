import os

import numpy as np
import torch
from tqdm import tqdm

from ..integrate import ScalableTSDFVolume
from ..integrate.utils.tsdf_ops import discrete2hash, hash2discrete
from .args import ProgramArgs


def build_graph(args: ProgramArgs):
    """
    This pipeline buid:
        border graph between diffferent patchs
        weighted connectivity graph from patch-mask correspondence
    """
    tsdf_volume = ScalableTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))

    kmeans_save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor + "_outlier_removed")
    kmeans_labels: torch.Tensor = torch.load(os.path.join(kmeans_save_dir, "labels.pt")).to(args.pipeline_device)

    # ! STEP 1: get border graph, whether two patchs are spacially attached to each other
    # get the neighbor hash
    x, y, z = np.meshgrid(np.array([0, -1, +1]), np.array([0, -1, +1]), np.array([0, -1, +1]), indexing="ij")
    relative_neighbors = torch.from_numpy(np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T[1:].reshape(1, -1, 3)).to(args.pipeline_device)  # 1, 26, 3

    discrete_coord = hash2discrete(tsdf_volume._voxel_hash)
    neighbor_hash = discrete2hash((discrete_coord.reshape(-1, 1, 3) + relative_neighbors))  # N, 26

    enlarged_hash_pool, inverse_idx = torch.cat([tsdf_volume._voxel_hash, neighbor_hash.reshape(-1)]).unique(sorted=True, return_inverse=True)

    # we construct a mapping to patches id, and get the patches id of neighbors
    patch_id_map = -1 + torch.zeros_like(enlarged_hash_pool)  # for neighbors outside we use -1
    patch_id_map[inverse_idx[: tsdf_volume._voxel_hash.shape[0]]] = kmeans_labels
    neighbor_patch_id = patch_id_map[inverse_idx[tsdf_volume._voxel_hash.shape[0] :]].reshape(neighbor_hash.shape)  # N

    have_different_neighbor = (neighbor_patch_id != -1) * (neighbor_patch_id != kmeans_labels.reshape(-1, 1))  # N, 26

    # build the graph
    temp = torch.argwhere(have_different_neighbor)

    self_id = kmeans_labels[temp[:, 0]]
    neighbor_id = neighbor_patch_id[temp[:, 0], temp[:, 1]]
    edge = self_id * (args.kmeans_cluster_num + 1) + neighbor_id
    border_counts = edge.bincount(minlength=(args.kmeans_cluster_num + 1) ** 2).reshape(args.kmeans_cluster_num + 1, args.kmeans_cluster_num + 1)
    # change unknown id 's bounder count to 0
    border_counts[0, :] = 0
    border_counts[:, 0] = 0

    # ! STEP 2: build weighted positive and negative connectivity graph
    # the weight have some basic options
    # 1. +/- 1
    # 2. num voxels in total for this patch
    # 2. num voxel in this frame / some kind of threshold
    # 3. num pixel of most likely  / total pixel

    kmeans_save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor)
    kmeans_labels: torch.Tensor = torch.load(os.path.join(kmeans_save_dir, "kmeans_labels.pt")).to(args.pipeline_device)

    voxel_sizes = kmeans_labels.bincount(minlength=args.kmeans_cluster_num + 1).cpu().numpy()
    voxel_size_conf = np.clip(voxel_sizes / args.graph_voxel_size_threshold, 0, 1)

    patch_corres_save_dir = os.path.join(args.save_dir, "patch_corres_ext-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor)

    positive_connectivity = np.zeros(shape=(args.kmeans_cluster_num + 1, args.kmeans_cluster_num + 1), dtype=float)
    negative_connectivity = np.zeros(shape=(args.kmeans_cluster_num + 1, args.kmeans_cluster_num + 1), dtype=float)

    for name in tqdm(os.listdir(patch_corres_save_dir)):
        frame_info = torch.load(os.path.join(patch_corres_save_dir, name))

        frame_mask_id = -1 + np.zeros(shape=(args.kmeans_cluster_num + 1,), dtype=int)
        frame_voxel_frac_conf = np.zeros(shape=(args.kmeans_cluster_num + 1,), dtype=float)
        frame_pixel_frac_conf = np.zeros(shape=(args.kmeans_cluster_num + 1,), dtype=float)
        num_pixel = frame_info["masks"].shape[0] * frame_info["masks"].shape[1]

        for corres in frame_info["patch_corres"]:
            idx = corres["patch_id"]
            frame_mask_id[idx] = corres["most_likely_maskid"]

            frame_voxel_frac_conf[idx] = np.clip(corres["num_voxel_in_frame"] / voxel_sizes[idx] / args.graph_voxel_frac_threshold, 0, 1)

            frame_pixel_frac_conf[idx] = np.clip(corres["num_pixel_in_mask"] / num_pixel / args.graph_pixel_frac_threshold, 0, 1)

        frame_weight = np.ones(shape=(args.kmeans_cluster_num + 1,), dtype=float)

        if "S" in args.graph_weight_method:
            frame_weight *= voxel_size_conf

        if "V" in args.graph_weight_method:
            frame_weight *= frame_voxel_frac_conf

        if "P" in args.graph_weight_method:
            frame_weight *= frame_pixel_frac_conf

        # for positive connectivity, we have to filter out mask_id = 0, as it means unclassified
        in_frame = (frame_mask_id != -1).reshape(-1, 1) * (frame_mask_id != -1).reshape(1, -1)
        not_background = (frame_mask_id != 0).reshape(-1, 1) * (frame_mask_id != 0).reshape(1, -1)
        edge_weight = frame_weight.reshape(-1, 1) * frame_weight.reshape(1, -1)

        same_mask = frame_mask_id.reshape(-1, 1) == frame_mask_id.reshape(1, -1)

        positive_connectivity += edge_weight * in_frame * not_background * same_mask
        negative_connectivity += edge_weight * in_frame * (~same_mask)

    # saving
    print("Saving")
    saving_name = args.graph_weight_method if args.graph_weight_method != "" else "binary"
    save_dir = os.path.join(args.save_dir, "connectivity_graph_ext-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor + "_" + saving_name)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "border_counts.npy"), border_counts.cpu().numpy())
    np.save(os.path.join(save_dir, "positive_connectivity.npy"), positive_connectivity)
    np.save(os.path.join(save_dir, "negative_connectivity.npy"), negative_connectivity)
