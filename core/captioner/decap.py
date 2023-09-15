import torch
import torch.nn.functional as f
from decap import DeCap

from .base import BaseCaptioner


class DeCapCaptioner(BaseCaptioner):
    decap: DeCap = None

    def __init__(
        self,
        decap_ckpt: str,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.decap_ckpt = decap_ckpt

    def load_model(self):
        self.decap = DeCap()
        self.decap.load_state_dict(torch.load(self.decap_ckpt), strict=False)
        self.decap = self.decap.eval().to(self.device)

    @torch.no_grad()
    def get_caption(self, feat: torch.Tensor, normalize: bool = True) -> str:
        """
        feat: a (512, ) dim vector
        """
        if normalize:
            feat = f.normalize(feat, dim=-1)

        return self.decap.decode(feat.to(self.device), entry_length=128).replace("<|startoftext|>", "").replace("<|endoftext|>", "")
