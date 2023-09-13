import os

import numpy as np
from tqdm import tqdm

from ..integrate import ScalableTSDFVolume
from .args import ProgramArgs
from .get_dataset import get_dataset


def tsdf(args: ProgramArgs):
    """
    this pipeline saves the tsdf volume and the corresponding vertices and mesh in {save_dir}/tsdf folder
    """
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

    print("Integrating TSDF...")
    for idx in tqdm(range(len(dataset["scan_dataset"]))[args.start : args.end : args.stride]):
        inputs = dataset["scan_dataset"].get_torch_tensor(
            idx,
            device=args.pipeline_device,
            keys={"depth", "depth_intr", "pose"},
        )

        tsdf_volume.integrate_tsdf(
            depth=inputs["depth"],
            depth_intr=inputs["depth_intr"],
            cam_pose=inputs["pose"],
        )
    print("Finished! Saving")

    save_dir = os.path.join(args.save_dir, "tsdf")
    os.makedirs(save_dir, exist_ok=True)
    if args.tsdf_unpruned:
        tsdf_volume.save(os.path.join(save_dir, "tsdf_volume_unpruned.pt"))

    # prune
    tsdf_volume.prune_voxel()
    tsdf_volume.save(os.path.join(save_dir, "tsdf_volume.pt"))

    # save vertices and faces
    verts, faces = tsdf_volume.extract_mesh()
    np.save(os.path.join(save_dir, "verts.npy"), verts)
    np.save(os.path.join(save_dir, "faces.npy"), faces)
