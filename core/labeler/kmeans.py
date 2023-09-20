from typing import Union

import numpy as np
import torch
from fast_pytorch_kmeans import KMeans

from .base import BaseLabeler


class KMeansLabeler(BaseLabeler):
    def __init__(
        self,
        K: int,
        device: str = "cpu",
    ) -> None:
        self.K = K
        self.device = device

        self.kmeans = KMeans(n_clusters=K, mode="euclidean", max_iter=2048, verbose=2)

    def feat_to_label(
        self,
        feats: Union[torch.Tensor, np.ndarray],
    ):
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        feats = feats.to(self.device)

        labels = self.kmeans.fit_predict(feats.reshape(-1, feats.shape[-1]))

        return labels.reshape(feats.shape[:-1])
