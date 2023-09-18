from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as f
from PIL import Image
from pycocotools import mask as mask_utils

from .utils import boolean_mask_to_random_feats, resize_feats, resize_masks


class BaseExtractor:
    name: str

    def load_model(self):
        raise NotImplementedError

    @torch.no_grad()
    def extract(self, image: Image.Image) -> Dict:
        """
        Abstraction for extracting features or instance segmentatino masks. Possible keys are:
            feats, masks, scores, phrases, boxes
        """
        raise NotImplementedError

    def save(self, results: Dict, pth: str):
        """
        saving everything to numpy, then use torch.save
        """
        save_dict = {}
        for key, value in results.items():
            new_value = value
            new_key = key
            if isinstance(value, torch.Tensor):
                new_value = value.detach().cpu().numpy()

            if key == "masks":
                new_value = mask_utils.encode(np.asfortranarray(new_value))
                new_key = "rle"

            save_dict[new_key] = new_value

        torch.save(save_dict, f=pth)

    def load(self, pth: str, device: str):
        file_dict = torch.load(f=pth)

        load_dict = {}
        for key, value in file_dict.items():
            new_value = value
            new_key = key

            if key == "rle":
                new_value = mask_utils.decode(new_value).astype(bool)
                new_key = "masks"

            if isinstance(new_value, np.ndarray):
                new_value = torch.from_numpy(new_value).to(device)

            load_dict[new_key] = new_value

        return load_dict

    def get_feats(
        self,
        results: Dict,
        output_height: int = None,
        output_width: int = None,
        device: str = None,
        normalize: bool = False,
        dtype: str = None,
    ) -> torch.Tensor:
        assert "feats" in results.keys()

        feats = results["feats"]
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        else:
            assert isinstance(feats, torch.Tensor)

        if normalize:
            feats = f.normalize(feats, dim=-1)

        if (output_height is not None) and (output_width is not None):
            feats = resize_feats(dense_feats=feats, H=output_height, W=output_width)

        if normalize:
            feats = f.normalize(feats, dim=-1)

        if device is not None:
            feats = feats.to(device=device)

        if dtype is not None:
            feats = feats.to(getattr(torch, dtype))

        return feats

    def get_masks(
        self,
        results: Dict,
        output_height: int = None,
        output_width: int = None,
        device: str = None,
    ) -> torch.Tensor:
        assert "masks" in results.keys()

        masks = results["masks"]
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        else:
            assert isinstance(masks, torch.Tensor)

        if (output_height is not None) and (output_width is not None):
            masks = resize_masks(masks=masks, H=output_height, W=output_width)

        if device is not None:
            masks = masks.to(device=device)

        return masks


class RandomFeatureExtractor(BaseExtractor):
    """
    Generate random features, where the feature is the same within a mask
    """

    feat_dim: int

    def get_feats(
        self,
        results: Dict,
        output_height: int = None,
        output_width: int = None,
        device: str = None,
        normalize: bool = False,
        dtype: str = None,
    ) -> torch.Tensor:
        assert "masks" in results.keys()

        feats = boolean_mask_to_random_feats(
            boolean_masks=results["masks"],
            device=device,
            feat_dim=self.feat_dim,
        )

        return super().get_feats({"feats": feats}, output_height, output_width, device, normalize, dtype)
