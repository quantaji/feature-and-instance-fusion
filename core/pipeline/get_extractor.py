from ..extractor import (
    BaseExtractor,
    ColorExtractor,
    ConceptFusionFeatureExtractor,
    GroundedSAMInstanceExtractor,
    GroundTruthInstanceExtractor,
    GroundTruthSemanticExtractor,
    LSegFeatureExtractor,
    MaskRCNNMaskExtractor,
    RandomFeatureExtractor,
    RandomGroundedSAMFeatureExtractor,
    RandomSAMFeatureExtractor,
    SAMMaskExtractor,
)
from .args import ProgramArgs


def get_extractor(args: ProgramArgs) -> BaseExtractor:
    if args.extractor == "color":
        return ColorExtractor(device=args.extractor_device)

    elif args.extractor == "conceptfusion":
        return ConceptFusionFeatureExtractor(
            sam_type=args.sam_type,
            sam_ckpt=args.sam_ckpt,
            clip_type=args.clip_type,
            clip_download_root=args.clip_download_root,
            weaken_background=(args.waken_factor < 1.0),
            weaken_factor=args.waken_factor,
            extend_ratio=args.extend_ratio,
            global_weight=args.global_weight,
            temperature=args.temperature,
            device=args.extractor_device,
        )
    elif args.extractor == "grounded_sam":
        return GroundedSAMInstanceExtractor(
            ram_ckpt=args.ram_ckpt,
            grounding_dino_config_pth=args.grounding_dino_config_pth,
            grounding_dino_ckpt=args.grounding_dino_ckpt,
            sam_hq_ckpt=args.sam_hq_ckpt,
            tag_set=args.grounded_sam_tag_set,
            device=args.extractor_device,
        )

    elif args.extractor == "lseg":
        return LSegFeatureExtractor(
            lseg_ckpt=args.lseg_ckpt,
            device=args.extractor_device,
        )

    elif args.extractor == "mask_rcnn":
        return MaskRCNNMaskExtractor(
            mask_rcnn_ckpt=args.mask_rcnn_ckpt,
            device=args.extractor_device,
        )

    elif args.extractor == "random":
        return RandomFeatureExtractor(
            feat_dim=args.feat_dim,
            device=args.extractor_device,
        )

    elif args.extractor == "random_grounded_sam":
        return RandomGroundedSAMFeatureExtractor(
            ram_ckpt=args.ram_ckpt,
            grounding_dino_config_pth=args.grounding_dino_config_pth,
            grounding_dino_ckpt=args.grounding_dino_ckpt,
            sam_hq_ckpt=args.sam_hq_ckpt,
            feat_dim=args.feat_dim,
            tag_set=args.grounded_sam_tag_set,
            device=args.extractor_device,
        )

    elif args.extractor == "random_sam":
        return RandomSAMFeatureExtractor(
            feat_dim=args.feat_dim,
            sam_type=args.sam_type,
            sam_ckpt=args.sam_ckpt,
            device=args.extractor_device,
        )

    elif args.extractor == "sam":
        return SAMMaskExtractor(
            sam_type=args.sam_type,
            sam_ckpt=args.sam_ckpt,
            device=args.extractor_device,
        )

    elif args.extractor == "gt_instance":
        return GroundTruthInstanceExtractor(
            background_id=args.gt_instance_background_id,
            device=args.extractor_device,
        )

    elif args.extractor == "gt_semantic":
        return GroundTruthSemanticExtractor(
            num_classes=args.gt_num_classes,
            device=args.extractor_device,
        )

    else:
        raise NotImplementedError
