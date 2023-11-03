import os

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from ..integrate import GuidedPanopticFusionScalabelTSDFVolume
from ..integrate.utils.misc import intrinsic_rescale
from .args import ProgramArgs
from .get_dataset import get_dataset
from .get_extractor import get_extractor


def tsdf_guided_panoptic(args: ProgramArgs):
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

            if args.extractor == "gt_instance":
                image = dataset["scan_dataset"][idx]["instance_img"]
            else:
                image = dataset["scan_dataset"][idx]["color_img"]
            results = extractor.extract(image)
            extractor.save(results, instance_pth)

        # delete the extractor to save gpu mem
        del extractor
        torch.cuda.empty_cache()

    # load tsdf volume
    tsdf_volume = GuidedPanopticFusionScalabelTSDFVolume(
        voxel_size=args.tsdf_voxel_size,
        sdf_trunc=args.tsdf_truc,
        margin=args.tsdf_margin,
        device=args.pipeline_device,
    )
    if args.tsdf_unpruned:
        tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume_unpruned.pt"))
    else:
        tsdf_volume.load(os.path.join(args.save_dir, "tsdf/tsdf_volume.pt"))
    tsdf_volume.reset_instance()

    graph_saving_name = args.graph_weight_method if args.graph_weight_method != "" else "binary"
    graph_save_dir = os.path.join(args.save_dir, "graph_connect_etx-" + args.extractor + "_kmeans-ext-" + args.kmeans_extractor + "_" + graph_saving_name)

    tsdf_volume.load_guidance_label(
        hash_pth=os.path.join(graph_save_dir, "label_voxel_hash.pt"),
        label_pth=os.path.join(graph_save_dir, "merged_labels.pt"),
    )

    # get extractor
    extractor = get_extractor(args=args)
    if not args.save_extraction:
        # use model hear
        extractor.load_model()

    save_dir = os.path.join(args.save_dir, "tsdf_guided_panoptic_" + args.extractor)
    os.makedirs(save_dir, exist_ok=True)

    if args.output_height is None or args.output_width is None:
        args.output_height = dataset["scan_dataset"].color_height
        args.output_width = dataset["scan_dataset"].color_width

    print("Performing TSDF guided panoptic fusion")
    for idx in tqdm(range(len(dataset["scan_dataset"]))[args.start : args.end : args.stride]):
        if args.save_extraction:
            instance_pth = os.path.join(extraction_save_dir, "{:>06d}.pt".format(idx))
            results = extractor.load(instance_pth, device=args.pipeline_device)
        else:
            if args.extractor == "gt_instance":
                image = dataset["scan_dataset"][idx]["instance_img"]
            else:
                image = dataset["scan_dataset"][idx]["color_img"]
            results = extractor.extract(image)

        masks = extractor.get_masks(
            results=results,
            output_height=args.output_height,
            output_width=args.output_width,
            device=args.pipeline_device,
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

        masks_intr = intrinsic_rescale(
            inputs["color_intr"],
            H_ori=inputs["color"].shape[0],
            W_ori=inputs["color"].shape[1],
            H_new=args.output_height,
            W_new=args.output_width,
        )

        corres = tsdf_volume.integrate_instance_with_existing_voxel_and_guidance(
            masks=masks,
            masks_intr=masks_intr,
            cam_pose=inputs["pose"],
            depth_type=args.tsdf_depth_type,
            depth_intr=inputs["depth_intr"],
            depth=inputs["depth"],
            Io2D_threshold=args.guided_panoptic_Io2D_threshold,
        )

        torch.save(corres, os.path.join(save_dir, "corres_{:>06d}.pt".format(idx)))

    tsdf_volume.save_instance(os.path.join(save_dir, "panoptic_labels.pt"))

    # save mesh for visualization
    color_set = np.random.random(size=(args.kmeans_cluster_num + 1, 3))
    color_set[0, :] = 0.0

    verts, faces = tsdf_volume.extract_mesh()
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh_labels = tsdf_volume.extract_label_on_grid(verts=verts, device=args.pipeline_device)[0]
    color = color_set[mesh_labels]
    mesh.vertex_colors = o3d.utility.Vector3dVector(color)

    o3d.io.write_triangle_mesh(filename=os.path.join(save_dir, "colored_mash.ply"), mesh=mesh)
