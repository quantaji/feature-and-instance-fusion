import os

import numpy as np
import open3d as o3d
import torch

from ..integrate import FeatureFusionScalableTSDFVolume, PanopticFusionScalableTSDFVolume
from ..integrate.utils.voxel_ops import discrete2world, hash2discrete
from ..labeler import KMeansLabeler
from .args import ProgramArgs


def kmeans(args: ProgramArgs):
    # load
    tsdf_volume = FeatureFusionScalableTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))
    tsdf_volume.load_feats(os.path.join(args.save_dir, "tsdf_feature_" + args.kmeans_extractor + "/feats.pt"))

    random_feats = tsdf_volume._feat  # N, D
    position = discrete2world(
        discrete_coord=hash2discrete(tsdf_volume._voxel_hash),
        voxel_size=tsdf_volume.voxel_size,
        voxel_origin=tsdf_volume._vol_origin,
    )  # N, 3

    feats = torch.cat([random_feats, args.kmeans_position_factor * position], dim=1)

    # kmeans
    labeler = KMeansLabeler(K=args.kmeans_cluster_num, device=args.pipeline_device)
    labels = labeler.feat_to_label(feats=feats).detach().cpu()

    # create a mesh of colored mask
    labeled_tsdf_volume = PanopticFusionScalableTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    labeled_tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))
    labeled_tsdf_volume._instance_label_num = args.kmeans_cluster_num + 1
    labeled_tsdf_volume._instance = labels  # zero for unknown
    labeled_tsdf_volume._instance_w_sum = labeled_tsdf_volume._instance_weight_init + torch.zeros(size=(labeled_tsdf_volume.num_voxel,), dtype=torch.float32, device=labeled_tsdf_volume.device)

    verts, faces = labeled_tsdf_volume.extract_mesh()
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    color_set = np.random.random(size=(args.kmeans_cluster_num, 3))
    mesh_labels = labeled_tsdf_volume.extract_label_on_grid(verts=verts, device=args.pipeline_device)[0]
    color = color_set[mesh_labels]

    mesh.vertex_colors = o3d.utility.Vector3dVector(color)

    # save
    save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(obj=labels, f=os.path.join(save_dir, "kmeans_labels.pt"))
    o3d.io.write_triangle_mesh(filename=os.path.join(save_dir, "colored_mash.ply"), mesh=mesh)
