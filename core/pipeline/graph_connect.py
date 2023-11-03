import os

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from scipy.sparse import csgraph, csr_matrix

from ..integrate import PanopticFusionScalableTSDFVolume
from .args import ProgramArgs


def graph_connect(args: ProgramArgs):
    """
    This pipeline buid:
        border graph between diffferent patchs
        weighted connectivity graph from patch-mask correspondence
    """

    kmeans_save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor + "_outlier_removed")
    kmeans_labels: torch.Tensor = torch.load(os.path.join(kmeans_save_dir, "labels.pt")).to(args.pipeline_device)

    graph_saving_name = args.graph_weight_method if args.graph_weight_method != "" else "binary"
    graph_save_dir = os.path.join(args.save_dir, "connectivity_graph_ext-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor + "_" + graph_saving_name)

    border_counts = np.load(os.path.join(graph_save_dir, "border_counts.npy"))
    positive_connectivity = np.load(os.path.join(graph_save_dir, "positive_connectivity.npy"))
    negative_connectivity = np.load(os.path.join(graph_save_dir, "negative_connectivity.npy"))

    log_ratio = np.log(1e-8 + positive_connectivity) - np.log(1e-8 + negative_connectivity)

    adjacency_matrix = (positive_connectivity > args.positive_threshold) * (log_ratio > args.log_ratio_threshold) * (border_counts > 0)

    sparse_matrix = csr_matrix(adjacency_matrix)
    _, labels = csgraph.connected_components(sparse_matrix)

    # make the unknown class the smallest
    labels[labels == labels[0]] = -1
    labels = torch.from_numpy(labels).unique(sorted=True, return_inverse=True)[1]

    # merge all patched that has no patch mapping
    counts = torch.zeros(size=(labels.unique().shape[0],), dtype=torch.int64)
    patch_corres_save_dir = os.path.join(args.save_dir, "patch_corres_ext-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor)
    for name in tqdm(os.listdir(patch_corres_save_dir)):
        frame_info = torch.load(os.path.join(patch_corres_save_dir, name))

        for corres in frame_info["patch_corres"]:
            if corres["most_likely_maskid"] != 0:
                counts[labels[corres["patch_id"]]] += 1

    labels[counts[labels] == 0] = 0
    labels = labels.unique(sorted=True, return_inverse=True)[1]

    voxel_labels = labels.to(device=args.pipeline_device, dtype=torch.int64)[kmeans_labels]
    # also add a unique to reduce total amount
    voxel_labels = voxel_labels.unique(sorted=True, return_inverse=True)[1].detach().cpu()

    save_dir = os.path.join(args.save_dir, "graph_connect_etx-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor + "_" + graph_saving_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(obj=voxel_labels, f=os.path.join(save_dir, "merged_labels.pt"))
    torch.save(obj=labels, f=os.path.join(save_dir, "patch_to_label_mapping.pt"))

    # also save a ply file
    labeled_tsdf_volume = PanopticFusionScalableTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    labeled_tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))
    labeled_tsdf_volume._instance_label_num = args.kmeans_cluster_num + 1
    labeled_tsdf_volume._instance = voxel_labels.to(args.pipeline_device)
    labeled_tsdf_volume._instance_w_sum = labeled_tsdf_volume._instance_weight_init + torch.zeros(size=(labeled_tsdf_volume.num_voxel,), dtype=torch.float32, device=labeled_tsdf_volume.device)

    verts, faces = labeled_tsdf_volume.extract_mesh()
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    color_set = np.random.random(size=(args.kmeans_cluster_num, 3))
    color_set[0, :] = 0.0
    mesh_labels = labeled_tsdf_volume.extract_label_on_grid(verts=verts, device=args.pipeline_device)[0]
    color = color_set[mesh_labels]

    mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_triangle_mesh(filename=os.path.join(save_dir, "colored_mash.ply"), mesh=mesh)
    
    # also save the voxel hash, since the guided panoptic fusion may use a different set of hash
    torch.save(obj=labeled_tsdf_volume._voxel_hash, f=os.path.join(save_dir, "label_voxel_hash.pt"))
