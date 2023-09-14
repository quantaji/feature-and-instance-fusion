from typing import List, Union

import matplotlib
import numpy as np
import torch

try:
    import open3d as o3d
except:
    o3d = None


class BaseLabeler:
    """
    Base labeler to convert embeddings to labels or colors
    """

    def feat_to_label(self, feat: torch.Tensor):
        pass

    def label_to_color(
        self,
        label: Union[torch.Tensor, np.ndarray],
        scheme: str = "rand",
        ext_scheme: Union[np.ndarray, List[List[int]], List[List[float]]] = None,
    ):
        """
        For a label of shape (SHAPE, ), return color in range [0, 1] of shape (SHAPE, 3), as ndarray.
        Scheme: whether "ext" (external) or "rand" (random)
        ext_scheme: the color scheme
        """
        assert scheme in ["ext", "rand"]

        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()

        else:
            assert isinstance(label, np.ndarray)

        assert "int" in str(label.dtype)

        if scheme == "ext":
            assert ext_scheme is not None
            color_scheme = ext_scheme

            if isinstance(color_scheme, list):
                color_scheme = np.array(color_scheme)

            if "int" in str(color_scheme.dtype):
                color_scheme = color_scheme.astype(float) / 255

        else:
            color_scheme = np.concatenate([np.random.random((label.max() + 1, 3))], axis=0)

        return color_scheme[label]

    def color_to_o3d_color(self, colors: np.ndarray):
        assert o3d is not None
        return o3d.utility.Vector3dVector(colors)

    def score_to_color(
        self,
        score: Union[torch.Tensor, np.ndarray],  # in [0, 1]
        scheme: str = "by",  # in "by" (blue to yellow) or "jet"
        normalize: bool = False,
        threshold: float = None,
    ):
        assert scheme in ["by", "jet"]

        if isinstance(score, torch.Tensor):
            score = score.detach().cpu().numpy()
        else:
            assert isinstance(score, np.ndarray)

        assert score.min() <= 1
        assert score.max() >= 0

        if normalize:
            score = (score - score.min()) / (score.max() - score.min() + 1e-12)

        if threshold is not None:
            score[score < threshold] = 0.0

        if scheme == "by":
            yellow = np.array([1.0, 1.0, 0.0]).reshape([1] * len(score.shape) + [3])
            blue = np.array([0.0, 0.0, 1.0]).reshape([1] * len(score.shape) + [3])

            score = np.expand_dims(score, axis=-1)

            color = score * yellow + (1 - score) * blue

        else:
            cmap = matplotlib.colormaps.get_cmap("jet")
            color = cmap(score)  # (SHAPE, 4)
            color = color.reshape(-1, 4)[:, :3].reshape(list(score.shape) + [3])

        return color
