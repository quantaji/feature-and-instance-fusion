from typing import Callable, Dict

import clip
import numpy as np
import torch
import torch.nn.functional as f
from clip.model import CLIP
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_hq, sam_model_registry
from segment_anything.modeling import Sam
from torch import Tensor
from torchvision import transforms

from .base import BaseExtractor
from .utils import resize_feats


class ConceptFusionFeatureExtractor(BaseExtractor):
    name = "conceptfusion"

    sam: Sam = None
    clip_model: CLIP = None
    clip_preprocessor: Callable = None
    mask_generator: SamAutomaticMaskGenerator = None

    def __init__(
        self,
        sam_type: str,
        sam_ckpt: str,
        clip_type: str = "ViT-B/32",
        clip_download_root: str = None,
        weaken_background: bool = False,
        weaken_factor: float = 1.0,
        extend_ratio: float = 0.0,
        global_weight: float = 1.0,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """
        weaken_background: whether to weaken the background and emphasize on the main object.
        weaken_factor: the factor that is beging multiplied to the "background" part of the image

        """

        # reference model
        self.sam_type = sam_type
        self.sam_ckpt = sam_ckpt
        self.clip_type = clip_type
        self.clip_download_root = clip_download_root

        self.weaken_background = weaken_background
        self.weaken_factor = weaken_factor
        self.extend_ratio = extend_ratio
        self.global_weight = global_weight
        self.temperature = temperature
        self.device = device

    def load_model(self):
        self.sam = sam_model_registry[self.sam_type](checkpoint=self.sam_ckpt).to(self.device).eval()
        self.clip_model, self.clip_preprocessor = clip.load(self.clip_type, download_root=self.clip_download_root)
        self.clip_model = self.clip_model.to(self.device).eval()

        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

    @torch.no_grad()
    def extract(self, image: Image) -> Dict:
        if self.mask_generator is None:
            self.load_model()

        image = image.convert("RGB")

        # Step 1: extract mask from SAM
        masks = self.mask_generator.generate(np.asarray(image))

        # Step 2: get clip features
        W, H = image.size
        regional_feats = []
        similarities = []

        # get global feats
        global_feat = f.normalize(self.clip_model.encode_image(self.clip_preprocessor(image).to(self.device).unsqueeze(0)).float(), dim=-1)

        for mask in masks:
            _input_bbox = ext_bbox(xywh_bbox=mask["bbox"], H=H, W=W, ext_ratio=self.extend_ratio)
            _input_img = crop_img(transforms.functional.to_tensor(image).to(self.device), xywh_bbox=_input_bbox)
            if self.weaken_background:
                _crop_mask = crop_img(torch.from_numpy(mask["segmentation"]), xywh_bbox=_input_bbox)
                _bg_idx = torch.argwhere((_crop_mask == False))
                _input_img[:, _bg_idx[:, 0], _bg_idx[:, 1]] *= self.weaken_factor
            _input_img = transforms.functional.to_pil_image(_input_img)
            _roi_img_tensor = self.clip_preprocessor(_input_img).unsqueeze(0).to(self.device)
            roi_feat = f.normalize(self.clip_model.encode_image(_roi_img_tensor).float(), dim=-1)

            regional_feats.append(roi_feat)

            cos_sim = f.cosine_similarity(global_feat, roi_feat, dim=-1)
            similarities.append(cos_sim)

        softmax_scores = f.softmax(torch.cat(similarities), dim=0)

        masks_np = np.stack([mask["segmentation"] for mask in masks], axis=0).astype(bool)
        boxes_np = np.stack([mask["bbox"] for mask in masks], axis=0)

        return {
            "masks": torch.from_numpy(masks_np),
            "boxes": torch.from_numpy(boxes_np),
            "global_feat": global_feat,
            "regional_feats": torch.stack(regional_feats, dim=0),
            "softmax_scores": softmax_scores,
            "H": H,
            "W": W,
        }

    @torch.no_grad()
    def get_feats(
        self,
        results: Dict,
        output_height: int = None,
        output_width: int = None,
        device: str = None,
        normalize: bool = True,
        dtype: str = None,
    ) -> Tensor:
        masks = results["masks"]
        global_feat = results["global_feat"]
        feat_dim = global_feat.shape[-1]
        H, W = results["H"], results["W"]

        dense_feats = torch.randn((H, W, feat_dim), device=device, dtype=torch.float32) * 1e-16

        for idx, roi_feats in enumerate(results["regional_feats"]):
            w = results["softmax_scores"][idx] * self.global_weight

            weighted_feat = f.normalize(w * global_feat + (1 - w) * roi_feats, dim=-1).squeeze(0)

            indices = torch.argwhere(masks[idx])

            dense_feats[indices[:, 0], indices[:, 1]] += weighted_feat.detach()

        return super().get_feats(
            results={"feats": dense_feats},
            output_height=output_height,
            output_width=output_width,
            device=device,
            normalize=normalize,
            dtype=dtype,
        )


def ext_bbox(xywh_bbox, H, W, ext_ratio):
    _x, _y, _w, _h = xywh_bbox

    x = np.clip(int(_x - _w * ext_ratio / 2), a_min=0, a_max=W)
    w = np.clip(int(_x + _w * (ext_ratio / 2 + 1)), a_min=0, a_max=W) - x

    y = np.clip(int(_y - _h * ext_ratio / 2), a_min=0, a_max=H)
    h = np.clip(int(_y + _h * (ext_ratio / 2 + 1)), a_min=0, a_max=H) - y

    return x, y, w, h


def crop_img(img, xywh_bbox):
    x, y, w, h = xywh_bbox
    return transforms.functional.crop(img, top=y, left=x, height=h, width=w)
