from typing import Callable, List, Union

import clip
import numpy as np
import torch
import torch.nn.functional as f
from clip.model import CLIP

from .base import BaseLabeler


class CLIPTextQuerier(BaseLabeler):
    """Get labels or scores from text"""

    model: CLIP = None
    preprocessor: Callable = None

    def __init__(
        self,
        clip_type: str = "ViT-B/32",
        device: str = "cpu",
        download_root: str = None,  # use default download root
    ) -> None:
        self.clip_type = clip_type
        self.device = device
        self.download_root = download_root

    def load_model(self):
        self.model, self.preprocessor = clip.load(
            self.clip_type,
            device=self.device,
            download_root=self.download_root,
        )
        self.model = self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: Union[List[str], str]):
        if self.model is None:
            self.load_model()
        return (
            f.normalize(
                self.model.encode_text(clip.tokenize(texts).to(self.device)),
                dim=-1,
            )
            .float()
            .cpu()
        )

    @torch.no_grad()
    def single_text_score(self, text: str, img_feats: torch.Tensor):
        """
        img_feats of shape (N1, N2, dim), txt_feats of shape (1, dim)
        """
        txt_feats = self.encode_text(texts=text)  # (1, 512)
        img_feats = img_feats.detach().float().cpu()  # (n1, n2, ..., 512)

        return torch.einsum("ik,jk->ij", img_feats.reshape(-1, img_feats.shape[-1]), txt_feats).reshape(img_feats.shape[:-1])

    @torch.no_grad()
    def single_text_others_score(
        self,
        text: str,
        img_feats: torch.Tensor,
        other_text: str = "others",  # this is the background text,
        temperature: float = 0.01,  # this controls the contrastiveness of the final score
    ):
        txt_feats = self.encode_text([text, other_text])
        img_feats = img_feats.detach().float().cpu()  # (n1, n2, ..., 512)

        sim = torch.einsum("ik,jk->ij", img_feats.reshape(-1, img_feats.shape[-1]), txt_feats).reshape(list(img_feats.shape[:-1]) + [2])  # (n1, n2, ..., 2)

        score = (sim / temperature).softmax(dim=-1).select(dim=-1, index=0)

        return score

    @torch.no_grad()
    def multi_text_query(
        self,
        texts: List[str],
        img_feats: torch.Tensor,
        other_text: str = None,  # if specified, add to text
    ):
        """
        Return the label, not scores
        """

        if other_text is not None:
            texts = texts + [other_text]

        txt_feats = self.encode_text(texts)
        img_feats = img_feats.detach().float().cpu()  # (n1, n2, ..., 512)

        sim: torch.Tensor = torch.einsum("ik,jk->ij", img_feats.reshape(-1, img_feats.shape[-1]), txt_feats).reshape(list(img_feats.shape[:-1]) + [txt_feats.shape[0]])  # (n1, n2, ...., n_txt)

        labels = sim.argmax(dim=-1)

        return labels
