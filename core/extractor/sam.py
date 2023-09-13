from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as f
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.modeling import Sam

from .base import BaseExtractor, RandomFeatureExtractor


class SAMMaskExtractor(BaseExtractor):
    name = "sam"

    sam: Sam
    mask_generator = SamAutomaticMaskGenerator

    def __init__(
        self,
        sam_type: str,
        sam_ckpt: str,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.sam_type = sam_type
        self.sam_ckpt = sam_ckpt

    def load_model(self):
        self.sam = sam_model_registry[self.sam_type](checkpoint=self.sam_ckpt).to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

    @torch.no_grad()
    def extract(self, image: Image) -> Dict:
        image = image.convert("RGB")

        # Step 1: extract mask from SAM
        masks = self.mask_generator.generate(np.asarray(image))
        masks = torch.from_numpy(np.stack([mask["segmentation"] for mask in masks], axis=0))

        return {"masks": masks}


class RandomSAMFeatureExtractor(RandomFeatureExtractor, SAMMaskExtractor):
    def __init__(
        self,
        feat_dim: int,
        sam_type: str,
        sam_ckpt: str,
        device: str = "cpu",
    ) -> None:
        self.feat_dim = feat_dim

        super().__init__(sam_type, sam_ckpt, device)
