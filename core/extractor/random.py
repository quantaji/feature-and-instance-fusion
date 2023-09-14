from typing import Dict

import torch
import torch.nn.functional as f
from PIL import Image

from .base import BaseExtractor


class RandomFeatureExtractor(BaseExtractor):
    name = "random"

    def __init__(
        self,
        feat_dim: int,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.feat_dim = feat_dim

    def load_model(self):
        pass

    @torch.no_grad()
    def extract(self, image: Image) -> Dict:
        return {}

    def get_feats(
        self,
        results: Dict,
        output_height: int = None,
        output_width: int = None,
        device: str = None,
        normalize: bool = False,
        dtype: str = None,
    ) -> torch.Tensor:
        assert output_height is not None
        assert output_width is not None

        tgt_device = device if device is not None else "cpu"
        tgt_dtype = getattr(torch, dtype) if dtype is not None else torch.float32

        feats = torch.randn(
            size=(output_height, output_width, self.feat_dim),
            dtype=tgt_dtype,
            device=tgt_device,
        )

        if normalize:
            feats = f.normalize(feats, dim=-1)

        return feats
