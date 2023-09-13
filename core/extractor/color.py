from typing import Dict

from PIL import Image
from torchvision.transforms.functional import to_tensor

from .base import BaseExtractor


class ColorExtractor(BaseExtractor):
    name = 'color'

    def __init__(
        self,
        device: str = "cpu",
    ) -> None:
        self.device = device

    def load_model(self):
        pass

    def extract(self, image: Image) -> Dict:
        return {
            "feats": to_tensor(image).movedim(0, -1),
        }
