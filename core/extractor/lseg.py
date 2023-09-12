from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as f
from lseg import LSegNet
from PIL import Image
from torchvision import transforms

from .base import BaseExtractor


class LSegFeatureExtractor(BaseExtractor):
    """
    Extract dense LSeg features.
    """

    lseg: LSegNet = None

    def __init__(self, lseg_ckpt: str, device: str = "cpu") -> None:
        self.lseg_ckpt = lseg_ckpt
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        self.trans = transforms.Compose(
            [
                transforms.Resize([480, 640]),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )
        self.device = device

    def load_model(self):
        lseg = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )
        lseg.load_state_dict(torch.load(self.lseg_ckpt))
        self.lseg = lseg.eval().to(self.device)

    @torch.no_grad()
    def extract(self, image: Image) -> Dict:
        if self.lseg is None:
            self.load_model()

        img_tsr = self.trans(image.convert("RGB")).unsqueeze(0).to(self.device)  # (H, W, dim)
        feats = self.lseg.forward(img_tsr).squeeze(0).movedim(0, -1)

        return {"feats": feats}

    @torch.no_grad()
    def get_feats(
        self,
        results: Dict,
        output_height: int = None,
        output_width: int = None,
        device: str = None,
        normalize: bool = True,  # LSeg we want to normalize it
    ) -> torch.Tensor:
        return super().get_feats(results, output_height, output_width, device, normalize)
