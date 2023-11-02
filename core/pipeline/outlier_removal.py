import os

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm, trange

from ..integrate import PanopticFusionScalableTSDFVolume
from ..integrate.utils.voxel_ops import discrete2world, hash2discrete
from .args import ProgramArgs


def outlier_removal(args: ProgramArgs):
    # load
    tsdf_volume = PanopticFusionScalableTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))

    world_coord = discrete2world(
        discrete_coord=hash2discrete(tsdf_volume._voxel_hash),
        voxel_size=tsdf_volume.voxel_size,
        voxel_origin=tsdf_volume._vol_origin,
    )  # N, 3

    load_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor)
    label = torch.load(f=os.path.join(load_dir, "kmeans_labels.pt")).to(args.pipeline_device)

    new_label = torch.zeros_like(label)

    for i in trange(args.kmeans_cluster_num):
        indices = torch.argwhere(label == i).reshape(-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_coord[indices].cpu().numpy())
        included_idx = pcd.remove_statistical_outlier(
            nb_neighbors=args.label_outlier_removal_nb_neighbors,
            std_ratio=args.label_outlier_removal_std_ratio,
            print_progress=False,
        )[1]
        new_label[indices[included_idx]] = i + 1
        # print(len(included_idx) / indices.shape[0])

    # fuse it into tsdf_volume
    tsdf_volume._instance_label_num = args.kmeans_cluster_num + 2
    tsdf_volume._instance = new_label
    tsdf_volume._instance_w_sum = tsdf_volume._instance_weight_init + torch.zeros(size=(tsdf_volume.num_voxel,), dtype=torch.float32, device=tsdf_volume.device)

    color_set = np.random.random(size=(args.kmeans_cluster_num + 1, 3))
    color_set[0, :] = 1.0
    # color_set[1:, :] = 0.0

    verts, faces = tsdf_volume.extract_mesh()
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh_labels = tsdf_volume.extract_label_on_grid(verts=verts, device=args.pipeline_device)[0]
    color = color_set[mesh_labels]
    mesh.vertex_colors = o3d.utility.Vector3dVector(color)

    save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor + "_outlier_removed")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(obj=new_label, f=os.path.join(save_dir, "labels.pt"))
    o3d.io.write_triangle_mesh(filename=os.path.join(save_dir, "colored_mash.ply"), mesh=mesh)
