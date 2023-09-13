from typing import List, Union

import numpy as np
import torch
from skimage import measure
from torch import Tensor

from .scalable_tsdf_volume import ScalableTSDFVolume
from .utils.tsdf_ops import filter_voxel_from_depth_and_image_and_get_pixels
from .utils.voxel_ops import discrete2hash, discrete2world, discretize_3d, hash2discrete, inhomo2homo
from .utils.voxel_ray_cast import voxel_ray_cast

try:
    import open3d as o3d
except:
    o3d = None


class FeatureFusionScalableTSDFVolume(ScalableTSDFVolume):
    """
    TSDF volume for feature extraction, used for color or CLIP embedding integration/fusion.
    """

    extensive_properties = [
        "_voxel_hash",
        "_tsdf",
        "_tsdf_w_sum",
        "_feat",
        "_feat_w_sum",
        "_feat_w2_sum",
        "_feat_square",
    ]

    variance_properties = ["_feat_w2_sum", "_feat_square"]

    _feat_dtype: torch.dtype
    _feat_dim: int
    _include_var: bool
    _feat_weight_init = 0.5  # this is to avoid division by zero in calculating variance

    _feat: Tensor
    _feat_w_sum: Tensor
    _feat_w2_sum: Tensor
    _feat_square: Tensor

    def reset_feature(self, dim: int, include_var: bool, dtype: torch.dtype):
        self._feat_dim = dim
        self._include_var = include_var
        self._feat_dtype = dtype

        self._feat = torch.zeros(size=(self.num_voxel, self._feat_dim), dtype=dtype, device=self.device)

        self._feat_w_sum = self._feat_weight_init + torch.zeros(size=(self.num_voxel,), dtype=torch.float32, device=self.device)

        if include_var:
            self._feat_square = torch.zeros(size=(self.num_voxel, self._feat_dim), dtype=dtype, device=self.device)

            self._feat_w2_sum = (self._feat_weight_init) ** 2 + torch.zeros(size=(self.num_voxel,), dtype=torch.float32, device=self.device)

        else:
            # if there is this two attribute delete it
            for attr in self.variance_properties:
                if hasattr(self, attr):
                    delattr(self, attr)

    def __init__(
        self,
        voxel_size: float,
        sdf_trunc: float,
        margin: float,
        feat_dim: int = 0,
        include_var: bool = False,
        feat_dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(voxel_size, sdf_trunc, margin, device)
        self.reset_feature(dim=feat_dim, include_var=include_var, dtype=feat_dtype)

    @property
    def feat_dim(self):
        return self._feat_dim

    def save_feats(self, pth: str):
        feats_obj = {
            "feat": self._feat,
            "feat_w_sum": self._feat_w_sum,
        }
        if self._include_var:
            feats_obj.update(
                {
                    "feat_square": self._feat_square,
                    "feat_w2_sum": self._feat_w2_sum,
                }
            )
        torch.save(obj=feats_obj, f=pth)

    def load_feats(self, pth: str):
        stat_dict = torch.load(pth)

        self._feat = stat_dict["feat"].to(self.device)
        self._feat_w_sum = stat_dict["feat_w_sum"].to(self.device)
        self._feat_dim = self._feat.shape[1]

        if "feat_square" in stat_dict.keys():
            self._include_var = True
            self._feat_square = stat_dict["feat_square"].to(self.device)
            self._feat_w2_sum = stat_dict["feat_w2_sum"].to(self.device)

    @torch.no_grad()
    def integrate_feature_with_exsisting_voxel(
        self,
        feat_img: Tensor,
        feat_intr: Tensor,
        cam_pose: Tensor,
        depth_type: str,
        depth_intr: Tensor = None,
        depth: Tensor = None,
        obs_weight: float = 1.0,
    ):
        """
        Integrade feature, with exsisting voxels. We assume depths and feature camera share same pose.
        depth_type: this arguments specify how depth information are extracted, in order to back project pixels to 3d to get pixel voxel correspondence
            'sensor': use depth image from real sensor, not so accurate
            'mesh_rc': ray casting of marching cube's extracted mesh, open3d is needed
            'voxel_rc' ray casting of voxel's cubes' mesh, if you install open3d, it also uses ray cast. If not, it use my self written torch functions, which is slow, about 400ms per frame. (or you can use voxel_rc_torch to manually test it)
        feat_img is of shape HxWxC
        """
        assert depth_intr.size() in [(3, 3), (4, 4)], "[!] `cam_intr' should be of shape (3, 3)."
        assert feat_intr.size() in [(3, 3), (4, 4)], "[!] `feat_intr' should be of shape (3, 3)."
        assert cam_pose.size() == (4, 4), "[!] `cam_pose' should be of shape (4, 4)."

        assert feat_img.size(2) == self.feat_dim, "[!] `dense_feat's third dimension should be equal to TSDF Volume's feature dimension."

        assert depth_type in {"sensor", "mesh_rc", "voxel_rc", "voxel_rc_torch"}

        # camera related
        H_f, W_f = feat_img.shape[0], feat_img.shape[1]

        # get depth
        device = self.device
        if depth_type == "sensor":
            assert depth_intr is not None
            assert depth is not None

        elif depth_type == "mesh_rc":
            assert o3d is not None, "install open3d 0.17.0 to enable this feature"

            depth = torch.from_numpy(
                self.get_depth_from_mc_mesh(
                    cam_intr=feat_intr,
                    cam_pose=cam_pose,
                    H=H_f,
                    W=W_f,
                )
            ).to(device)
            depth_intr = feat_intr

        elif depth_type == "voxel_rc" and o3d is not None:
            depth = torch.from_numpy(
                self.get_depth_from_mc_voxel(
                    cam_intr=feat_intr,
                    cam_pose=cam_pose,
                    H=H_f,
                    W=W_f,
                )
            ).to(device)
            depth_intr = feat_intr
            depth[depth > 0] += self.voxel_size * 0.01

        else:
            depth_intr = feat_intr
            _, _, _, depth = voxel_ray_cast(
                voxel_hash=self._voxel_hash,
                voxel_origin=self._vol_origin,
                voxel_size=self.voxel_size,
                cam_intr=feat_intr,
                cam_pose=cam_pose,
                H=H_f,
                W=W_f,
            )

        # get valid voxel, depth
        indices, pix_x, pix_y = filter_voxel_from_depth_and_image_and_get_pixels(
            world_c=inhomo2homo(
                discrete2world(
                    hash2discrete(self._voxel_hash),
                    voxel_size=self.voxel_size,
                    voxel_origin=self._vol_origin,
                )
            ),
            cam_pose=cam_pose,
            depth=depth,
            depth_intr=depth_intr,
            feat_intr=feat_intr,
            H_f=H_f,
            W_f=W_f,
            margin=self._margin,
        )
        if indices is None:
            return
        feat_obs = feat_img[pix_y, pix_x]

        del depth, pix_x, pix_y

        # step 2: merge feature
        # Step 2.3: forge feature
        feat_old = self._feat[indices]
        feat_w_old = self._feat_w_sum[indices]

        feat_w_new = feat_w_old + obs_weight
        feat_new = feat_old + obs_weight * (feat_obs - feat_old) / feat_w_new.unsqueeze(-1)

        # # forge
        self._feat[indices] = feat_new.to(self._feat_dtype)
        self._feat_w_sum[indices] = feat_w_new

        del feat_w_old, feat_w_new

        if self._include_var:
            feat_w2_old = self._feat_w2_sum[indices]
            feat_square_old = self._feat_square[indices]

            feat_w2_new = feat_w2_old + obs_weight**2
            feat_square_new = feat_square_old + obs_weight * (feat_obs - feat_old) * (feat_obs - feat_new)

            self._feat_square[indices] = feat_square_new.to(self._feat_dtype)
            self._feat_w2_sum[indices] = feat_w2_new

            del feat_w2_old, feat_square_old, feat_w2_new, feat_square_new, feat_obs

    @torch.no_grad()
    def extract_feat_on_grid(self, verts: np.ndarray, device=None):
        if device is None:
            device = self.device

        # extract colors, dense features is very expensive, so I choose to use the sparse array
        hash_selected = discrete2hash(discretize_3d(torch.from_numpy(verts.copy()).to(device), voxel_size=self._voxel_size, voxel_origin=self._vol_origin.to(device)))

        hash_new, indices = torch.cat([self._voxel_hash.to(device), hash_selected], dim=0).unique(return_inverse=True)
        idx_old, idx_selected = indices[: self.num_voxel], indices[self.num_voxel :]
        new_n_vox = hash_new.size(0)

        if self.num_voxel < new_n_vox:
            feat_merge = torch.zeros(size=(new_n_vox, self._feat_dim), dtype=torch.float32, device=device)
            feat_merge[idx_old] = self._feat.to(device)

            feat_w_sum_merge = self._feat_weight_init + torch.zeros(size=(new_n_vox,), dtype=torch.float32, device=device)
            feat_w_sum_merge[idx_old] = self._feat_w_sum.to(device)

            if self._include_var:
                feat_sqare_merge = torch.zeros(size=(new_n_vox, self._feat_dim), dtype=torch.float32, device=device)
                feat_sqare_merge[idx_old] = self._feat_square.to(device)

                feat_w2_sum_merge = (self._feat_weight_init) ** 2 + torch.zeros(size=(new_n_vox,), dtype=torch.float32, device=device)
                feat_w2_sum_merge[idx_old] = self._feat_w2_sum.to(device)
        else:
            feat_merge = self._feat.to(device)
            feat_w_sum_merge = self._feat_w_sum.to(device)
            if self._include_var:
                feat_sqare_merge = self._feat_square.to(device)
                feat_w2_sum_merge = self._feat_w2_sum.to(device)

        feats = feat_merge[idx_selected].cpu().numpy()
        feats_w_sum = feat_w_sum_merge[idx_selected].cpu().numpy()
        if self._include_var:
            feats_sqare = feat_sqare_merge[idx_selected].cpu().numpy()
            feats_w2_sum = feat_w2_sum_merge[idx_selected].cpu().numpy()

        del hash_selected, idx_old, idx_selected, feat_merge, hash_new, indices, feat_w_sum_merge
        if self._include_var:
            del feat_sqare_merge, feat_w2_sum_merge
        torch.cuda.empty_cache()

        if self._include_var:
            return feats, feats_w_sum, feats_sqare, feats_w2_sum
        else:
            return feats, feats_w_sum
