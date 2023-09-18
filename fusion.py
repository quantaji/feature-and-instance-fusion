import tyro

from core.pipeline.args import ProgramArgs
from core.pipeline.gradslam_feature import gradslam_feature
from core.pipeline.kmeans import kmeans
from core.pipeline.patch_corres import patch_corres
from core.pipeline.tsdf import tsdf
from core.pipeline.tsdf_feature import tsdf_feature
from core.pipeline.tsdf_panoptic import tsdf_panoptic
from core.pipeline.build_graph import build_graph

if __name__ == "__main__":
    args = tyro.cli(ProgramArgs)

    if args.pipeline == "tsdf":
        tsdf(args=args)

    elif args.pipeline == "tsdf_feature":
        tsdf_feature(args=args)

    elif args.pipeline == "tsdf_panoptic":
        tsdf_panoptic(args=args)

    elif args.pipeline == "gradslam_feature":
        gradslam_feature(args=args)

    elif args.pipeline == "kmeans":
        kmeans(args=args)

    elif args.pipeline == "patch_corres":
        patch_corres(args=args)

    elif args.pipeline == "build_graph":
        build_graph(args=args)

    else:
        raise NotImplementedError
