import torch
import torch.nn.functional as f
from clipcap import ClipCaptionModel, generate2
from transformers import GPT2Tokenizer

from .base import BaseCaptioner


class ClipCapCaptioner(BaseCaptioner):
    clipcap: ClipCaptionModel = None

    def __init__(
        self,
        clipcap_ckpt: str,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.clipcap_ckpt = clipcap_ckpt

    def load_model(self):
        self.clipcap = ClipCaptionModel(
            prefix_length=40,
            clip_length=40,
            prefix_size=512,
            num_layers=8,
            mapping_type="transformer",
        )
        self.clipcap.load_state_dict(torch.load(self.clipcap_ckpt, map_location="cpu"), strict=False)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.clipcap = self.clipcap.eval().to(self.device)

    @torch.no_grad
    def get_caption(self, feat: torch.Tensor, normalize: bool = True) -> str:
        if normalize:
            feat = f.normalize(feat, dim=-1)

        prefix_embed = self.clipcap.clip_project(feat.to(self.device)).reshape(1, self.clipcap.prefix_length, -1)
        generated_text = generate2(
            self.clipcap,
            self.tokenizer,
            embed=prefix_embed,
        )
        return generated_text
