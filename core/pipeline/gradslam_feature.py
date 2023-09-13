import os

import torch
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages
from tqdm import tqdm

from ..extractor.utils import resize_feats
from ..integrate.utils.misc import intrinsic_rescale
from .args import ProgramArgs
from .get_dataset import get_dataset
from .get_extractor import get_extractor


def gradslam_feature(args: ProgramArgs):
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

    slam = PointFusion(odom="gt", dsratio=1, device=args.pipeline_device, use_embeddings=True)
    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(device=args.pipeline_device)
    # start fusion
    extractor = get_extractor(args=args)
    if not args.save_extraction:
        # use model hear
        extractor.load_model()

    # in gradslam we have to make everything of same shape
    if args.output_height is None or args.output_width is None:
        args.output_height = dataset["scan_dataset"].depth_height
        args.output_width = dataset["scan_dataset"].depth_width

    print("Running PointFusion (incremental mode)...")
    for idx in tqdm(range(len(dataset["scan_dataset"]))[args.start : args.end : args.stride]):
        torch.cuda.empty_cache()

        if args.save_extraction:
            instance_pth = os.path.join(extraction_save_dir, "{:>06d}.pt".format(idx))
            results = extractor.load(instance_pth, device=args.pipeline_device)
        else:
            if args.extractor == "gt_semantic":
                image = dataset["scan_dataset"][idx]["label_img"]
            else:
                image = dataset["scan_dataset"][idx]["color_img"]

            results = extractor.extract(image)

        inputs = dataset["scan_dataset"].get_torch_tensor(
            idx,
            device=args.pipeline_device,
            keys={
                "color",
                "color_intr",
                "depth",
                "depth_intr",
                "pose",
            },
        )

        feats = (
            extractor.get_feats(
                results=results,
                output_height=args.output_height,
                output_width=args.output_width,
                device=args.pipeline_device,
                dtype=args.feature_dtype,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        color = resize_feats(inputs["color"], H=args.output_height, W=args.output_width).unsqueeze(0).unsqueeze(0)

        depth = resize_feats(inputs["depth"].unsqueeze(-1), H=args.output_height, W=args.output_width).unsqueeze(0).unsqueeze(0)

        intr = (
            intrinsic_rescale(
                inputs["depth_intr"],
                H_ori=inputs["depth"].shape[0],
                W_ori=inputs["depth"].shape[1],
                H_new=args.output_height,
                W_new=args.output_width,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        pose = inputs["pose"].unsqueeze(0).unsqueeze(0)

        frame_cur = RGBDImages(
            rgb_image=color,
            depth_image=depth,
            intrinsics=intr,
            poses=pose,
            embeddings=feats,
        )

        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev, inplace=True)

    print("Finished! Saving")
    save_dir = os.path.join(args.save_dir, "gradslam_feature_" + args.extractor)
    os.makedirs(save_dir, exist_ok=True)
    pointclouds.save_to_h5(save_dir)
