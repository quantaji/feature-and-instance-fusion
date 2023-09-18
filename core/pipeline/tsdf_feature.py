import os

import torch
from tqdm import tqdm

from ..integrate import FeatureFusionScalableTSDFVolume
from ..integrate.utils.misc import intrinsic_rescale
from .args import ProgramArgs
from .get_dataset import get_dataset
from .get_extractor import get_extractor


def tsdf_feature(args: ProgramArgs):
    dataset = get_dataset(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        scan_id=args.scan_id,
    )

    # get extractor
    # first if we use save extraction, we first perform loop over all extraction and save them to the target folder
    if args.save_extraction:
        extractor = get_extractor(args=args)
        extractor.load_model()

        extraction_save_dir = os.path.join(args.save_dir, extractor.name)
        os.makedirs(extraction_save_dir, exist_ok=True)
        print("Extracting...")

        for idx in tqdm(range(len(dataset["scan_dataset"]))[args.start : args.end : args.stride]):
            instance_pth = os.path.join(extraction_save_dir, "{:>06d}.pt".format(idx))
            if os.path.exists(instance_pth):
                continue

            if args.extractor == "gt_semantic":
                image = dataset["scan_dataset"][idx]["label_img"]
            else:
                image = dataset["scan_dataset"][idx]["color_img"]

            results = extractor.extract(image)
            extractor.save(results, instance_pth)

        # delete the extractor to save gpu mem
        del extractor
        torch.cuda.empty_cache()

    # load tsdf volume
    tsdf_volume = FeatureFusionScalableTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    if args.tsdf_unpruned:
        tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume_unpruned.pt"))
    else:
        tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))

    # reset feature
    tsdf_volume.reset_feature(
        dim=args.feat_dim,
        include_var=args.tsdf_feat_include_var,
        dtype=getattr(torch, args.feature_dtype) if args.feature_dtype is not None else None,
    )

    # start fusion
    extractor = get_extractor(args=args)
    if not args.save_extraction:
        # use model hear
        extractor.load_model()

    if args.output_height is None or args.output_width is None:
        args.output_height = dataset["scan_dataset"].color_height
        args.output_width = dataset["scan_dataset"].color_width

    print("Performing TSDF feature fusion")
    for idx in tqdm(range(len(dataset["scan_dataset"]))[args.start : args.end : args.stride]):
        if args.save_extraction:
            instance_pth = os.path.join(extraction_save_dir, "{:>06d}.pt".format(idx))
            results = extractor.load(instance_pth, device=args.pipeline_device)

        else:
            if args.extractor == "gt_semantic":
                image = dataset["scan_dataset"][idx]["label_img"]
            else:
                image = dataset["scan_dataset"][idx]["color_img"]

            results = extractor.extract(image)

        feats = extractor.get_feats(
            results=results,
            output_height=args.output_height,
            output_width=args.output_width,
            device=args.pipeline_device,
            dtype=args.feature_dtype,
        )

        inputs = dataset["scan_dataset"].get_torch_tensor(
            idx,
            device=args.pipeline_device,
            keys={
                "depth",
                "depth_intr",
                "pose",
                "color",
                "color_intr",
            },
        )

        feat_intr = intrinsic_rescale(
            inputs["color_intr"],
            H_ori=inputs["color"].shape[0],
            W_ori=inputs["color"].shape[1],
            H_new=args.output_height,
            W_new=args.output_width,
        )

        tsdf_volume.integrate_feature_with_exsisting_voxel(
            feat_img=feats,
            feat_intr=feat_intr,
            depth_type=args.tsdf_depth_type,
            depth=inputs["depth"],
            depth_intr=inputs["depth_intr"],
            cam_pose=inputs["pose"],
        )

    print("Finished! Saving")
    save_dir = os.path.join(args.save_dir, "tsdf_feature_" + args.extractor)
    os.makedirs(save_dir, exist_ok=True)
    tsdf_volume.save_feats(os.path.join(save_dir, "feats.pt"))
