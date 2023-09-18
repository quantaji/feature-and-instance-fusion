import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProgramArgs:
    # dataset
    dataset_name: str = "ScanNet"
    dataset_dir: str = str(Path.home() / "Datasets/ScanNet")
    scan_id: str = "scene0000_00"

    # extractor
    extractor: str = "color"  # ["color", "conceptfusion", "grounded_sam", "lseg", "mask_rcnn",  "random_grounded_sam", "random_sam", "sam", "gt_instance", "gt_semantic"]
    extractor_device: str = "cpu"
    save_extraction: bool = False
    output_height: int = None  # none means unchanged
    output_width: int = None

    # extractor related
    ## SAM
    sam_ckpt: str = str(Path.home() / "Models/SAM/sam_vit_h_4b8939.pth")
    sam_type: str = "vit_h"
    # conceptfusion related
    clip_type: str = "ViT-B/32"  # this is the clip model used by clip-glass and DeCap
    clip_download_root: str = str(os.environ.get("SCRATCH", Path.home()) / Path(".cache/clip"))
    waken_factor: float = 1.0
    global_weight: float = 1.0
    temperature: float = 1.0
    extend_ratio: float = 0.0
    # lseg
    lseg_ckpt: str = str(Path.home() / "Models/LSeg/lseg_minimal_e200.ckpt")
    # grounded-sam related
    grounding_dino_config_pth: str = "./config/grounding_dino_config/GroundingDINO_SwinT_OGC.py"
    grounding_dino_ckpt: str = str(os.environ.get("SCRATCH", Path.home())  / Path("Models/GroundingDINO/groundingdino_swint_ogc.pth"))
    ram_ckpt: str = str(os.environ.get("SCRATCH", Path.home())  / Path("Models/RAM_Tag2Text/ram_swin_large_14m.pth"))
    sam_hq_ckpt: str = str(os.environ.get("SCRATCH", Path.home())  / Path("Models/SAM_HQ/sam_hq_vit_h.pth"))
    # mask rcnn
    mask_rcnn_ckpt: str = None  # none means use default
    # random feature related
    feat_dim: int = 512
    # ground truth
    gt_num_classes: int = 41
    gt_instance_background_id: int = 0

    # pipeline related
    pipeline: str = "tsdf"  # ["tsdf", "tsdf_feature", "tsdf_panoptic", "tsdf_guided_panoptic", "gradslam_feature", "kmeans", "connectivity_graph", "kmeans"]
    ## If a pipeline use 2D frames, then this is needed:
    start: int = 0
    end: int = -1
    stride: int = 1
    pipeline_device: str = "cpu"  # device used for this pipeline
    ## If the pipeline needs tsdf this is related
    tsdf_voxel_size: float = 0.015
    tsdf_truc: float = 0.075
    tsdf_margin: float = 0.08
    tsdf_unpruned: bool = False  # whether to save or use the unpruned tsdf volume, used for some vertices query.
    tsdf_depth_type: str = "sensor"  # ["sensor", "mesh_rc", "voxel_rc", "voxel_rc_torch"] used in feature fusion and panoptic fusion
    ## if it is feature related, specify feature dtype
    feature_dtype: str = None  # "float" for torch.float32, "half" for torch.float16, use method getattr(torch, feature_dtype)
    tsdf_feat_include_var: bool = False  # whether to include variance calculation

    # save
    save_dir: str = str(os.environ.get("SCRATCH", Path.home()) / Path("Experiments/feat-and-instance-fusion"))

    # panoptic fusion config
    panoptic_threshold: float = 0.25

    # kmeans related
    kmeans_cluster_num: int = 512
    kmeans_extractor: str = 'random_grounded_sam'
