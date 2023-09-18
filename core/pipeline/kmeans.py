import os

from ..integrate import FeatureFusionScalableTSDFVolume
from .args import ProgramArgs
from ..labeler import KMeansLabeler


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

    # kmeans
    labeler = KMeansLabeler(K=args.kmeans_cluster_num, device=args.pipeline_device)
    labels = labeler.feat_to_label(feats=tsdf_volume._feat).detach().cpu()
-+
    # save
    save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(obj=labels, f=os.path.join(save_dir, "kmeans_labels.pt"))
