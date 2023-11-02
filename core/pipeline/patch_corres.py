import os

import torch
from tqdm import tqdm

from ..extractor.utils import isolate_boolean_masks
from ..integrate import ScalableTSDFVolume
from ..integrate.utils.misc import intrinsic_rescale
from ..integrate.utils.tsdf_ops import discrete2world, hash2discrete, inhomo2homo, pixel_voxel_corres_given_depth
from .args import ProgramArgs
from .get_dataset import get_dataset
from .get_extractor import get_extractor


def patch_corres(args: ProgramArgs):
    dataset = get_dataset(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        scan_id=args.scan_id,
    )

    tsdf_volume = ScalableTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))

    extractor = get_extractor(args=args)
    extraction_save_dir = os.path.join(args.save_dir, extractor.name)

    kmeans_save_dir = os.path.join(args.save_dir, "kmeans_" + args.kmeans_extractor + "_outlier_removed")
    kmeans_labels: torch.Tensor = torch.load(os.path.join(kmeans_save_dir, "labels.pt")).to(args.pipeline_device)

    save_dir = os.path.join(args.save_dir, "patch_corres_ext-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor)
    os.makedirs(save_dir, exist_ok=True)

    if args.output_height is None or args.output_width is None:
        args.output_height = dataset["scan_dataset"].color_height
        args.output_width = dataset["scan_dataset"].color_width

    print("Performing KMeans Patch and 2D Segmentations correspondence calculation...")

    # we assume the features is already calculated, so we only read them from file
    for idx in tqdm(range(len(dataset["scan_dataset"]))[args.start : args.end : args.stride]):
        inputs = dataset["scan_dataset"].get_torch_tensor(
            idx,
            device=args.pipeline_device,
            keys={
                "pose",
                "color",
                "color_intr",
            },
        )

        cam_pose = inputs["pose"]

        masks_intr = intrinsic_rescale(
            inputs["color_intr"],
            H_ori=inputs["color"].shape[0],
            W_ori=inputs["color"].shape[1],
            H_new=args.output_height,
            W_new=args.output_width,
        )

        instance_pth = os.path.join(extraction_save_dir, "{:>06d}.pt".format(idx))
        results = extractor.load(instance_pth, device=args.pipeline_device)
        masks = extractor.get_masks(
            results=results,
            output_height=args.output_height,
            output_width=args.output_width,
            device=args.pipeline_device,
        )

        depth = torch.from_numpy(
            tsdf_volume.get_depth_from_mc_mesh(
                cam_intr=masks_intr,
                cam_pose=cam_pose,
                H=args.output_height,
                W=args.output_width,
            )
        ).to(args.pipeline_device)

        # map pixel to voxel
        px, py, pix_vox_idx = pixel_voxel_corres_given_depth(
            voxel_hash=tsdf_volume._voxel_hash,
            voxel_origin=tsdf_volume._vol_origin,
            voxel_size=tsdf_volume.voxel_size,
            cam_intr=masks_intr,
            cam_pose=cam_pose,
            H=args.output_height,
            W=args.output_width,
            depth=depth,
        )

        # convert masks to none intersecting masks
        # ! note that maskid = 0 means unrecognized, means background
        non_intersect_masks = torch.from_numpy(isolate_boolean_masks(masks.cpu().numpy())).to(masks.device)
        pix_patch_id = kmeans_labels[pix_vox_idx]
        pix_mask_id = non_intersect_masks[py, px]

        # We need the following statistics
        # 1. patch id
        # 2. number of voxels of this patch appears in this frame
        # 3. number of pixels that this voxel patch takes in this frame
        # 4. given all masks that share intersection with this voxel patch, what is the most propbable mask that have most pixel fit
        # 5. number of pixels for most probable masks and the current voxel patch

        patch_corres = []
        for i in range(1, args.kmeans_cluster_num + 1):
            if (pix_patch_id == i).count_nonzero() > 0:
                filt = pix_patch_id == i

                unique_maskid_in_this_patch, maskid_counts = pix_mask_id[filt].unique(sorted=True, return_counts=True)

                num_voxels_in_frame = pix_vox_idx[filt].unique(sorted=True).shape[0]

                num_pixels_in_frame = filt.count_nonzero().item()

                most_likely_maskid = unique_maskid_in_this_patch[maskid_counts.argmax()].item()

                num_pixels_of_most_likely_masks = maskid_counts.max().item()

                patch_corres.append(
                    {
                        "patch_id": i,
                        "num_voxel_in_frame": num_voxels_in_frame,
                        "num_pixel_in_frame": num_pixels_in_frame,
                        "most_likely_maskid": most_likely_maskid,
                        "num_pixel_in_mask": num_pixels_of_most_likely_masks,
                    }
                )

        frame_info = {
            "masks": non_intersect_masks.detach().cpu().numpy(),
            "num_masks": non_intersect_masks.max().item() + 1,
            "patch_corres": patch_corres,
        }

        torch.save(frame_info, os.path.join(save_dir, "{:>06d}.pt".format(idx)))
