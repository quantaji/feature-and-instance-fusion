from typing import Dict

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_tensor

from ..integrate.utils.misc import instance_id_to_one_hot_mask
from .base import BaseExtractor


class GroundTruthInstanceExtractor(BaseExtractor):
    name = "gt_instance"

    def __init__(
        self,
        background_id: int = -1,  # -1 means no background, usually
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.background_id = background_id

    def load_model(self):
        pass

    def extract(self, image: Image) -> Dict:
        """
        The input image is the gorund truth instance segmentaiton
        """

        instance = pil_to_tensor(image).squeeze(0)
        masks = instance_id_to_one_hot_mask(
            instance.long(),
            background_id=self.background_id,
        ).to(self.device)

        return {
            "masks": masks,
        }


class GroundTruthSemanticExtractor(BaseExtractor):
    """
    Fusing one hot semantic features
    """

    name = "gt_semantic"

    def __init__(
        self,
        num_classes: int = 41,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_classes = num_classes

    def load_model(self):
        pass

    def extract(self, image: Image) -> Dict:
        """
        The input image is the gorund truth instance segmentaiton
        """

        label = pil_to_tensor(image).squeeze(0).to(self.device)
        feats = torch.nn.functional.one_hot(label.long(), num_classes=self.num_classes)

        return {
            "feats": feats,
        }
