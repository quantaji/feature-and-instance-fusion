import numpy as np
import torch
from torch import Tensor

from .scalable_tsdf_volume import ScalableTSDFVolume
from .utils.panoptic_fusion_ops import instance_2d_map_3d
from .utils.tsdf_ops import filter_voxel_from_depth_and_image_and_get_pixels, pixel_voxel_corres_given_depth
from .utils.voxel_ops import discrete2hash, discrete2world, discretize_3d, hash2discrete, inhomo2homo
from .utils.voxel_ray_cast import voxel_ray_cast

try:
    import open3d as o3d
except:
    o3d = None


class PanopticFusionScalableTSDFVolume(ScalableTSDFVolume):
    """
    TSDF volume objects for panoptic fusion
    """

    extensive_properties = [
        "_voxel_hash",
        "_tsdf",
        "_tsdf_w_sum",
        "_instance",
        "_instance_w_sum",
    ]

    # instance related
    _instance_weight_init: float = 0.5
    _instance: Tensor
    _instance_w_sum: Tensor
    _instance_label_num: int

    def reset_instance(self):
        """
        the rule is:
            mask_id = 0 means unset, unclassified
            mask_id > 1 means an instance is assigned to this voxel
        """
        self._instance_label_num = 1
        self._instance = torch.zeros(size=(self.num_voxel,), dtype=torch.int64, device=self.device)
        self._instance_w_sum = self._instance_weight_init + torch.zeros(size=(self.num_voxel,), dtype=torch.float32, device=self.device)

    def __init__(
        self,
        voxel_size: float,
        sdf_trunc: float,
        margin: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(voxel_size, sdf_trunc, margin, device)
        self.reset_instance()

    @torch.no_grad()
    def integrate_instance_with_existing_voxel(
        self,
        masks: Tensor,
        masks_intr: Tensor,
        cam_pose: Tensor,
        threshold: float,
        depth_type: str,
        depth_intr: Tensor = None,
        depth: Tensor = None,
        obs_weight: float = 1.0,
    ):
        """
        masks: Tensor of shape (n_masks, H, W)
        naming rules:
            masks: one-hot version of instances labels
            (instance)_labels: none-one-hot version
        depth_type: this arguments specify how depth information are extracted, in order to back project pixels to 3d to get pixel voxel correspondence
            'sensor': use depth image from real sensor, not so accurate
            'mesh_rc': ray casting of marching cube's extracted mesh, open3d is needed
            'voxel_rc' ray casting of voxel's cubes' mesh, if you install open3d, it also uses ray cast. If not, it use my self written torch functions, which is slow, about 400ms per frame. (or you can use voxel_rc_torch to manually test it)
        Returns:
            the correspondence between 2D instance ids and 3D instance ids
        """
        if masks.shape[0] == 0:
            return

        device = self.device
        if depth_intr is not None:
            assert depth_intr.size() in [(3, 3), (4, 4)], "[!] `cam_intr' should be of shape (3, 3)."
        assert masks_intr.size() in [(3, 3), (4, 4)], "[!] `feat_intr' should be of shape (3, 3)."
        assert cam_pose.size() == (4, 4), "[!] `cam_pose' should be of shape (4, 4)."
        assert depth_type in {"sensor", "mesh_rc", "voxel_rc", "voxel_rc_torch"}

        # ! STEP 1, get depth and pixel corres
        H_m, W_m = masks.shape[-2], masks.shape[-1]

        layer_px: Tensor
        layer_py: Tensor
        layer_vol_idx: Tensor

        if depth_type == "sensor":
            assert depth_intr is not None
            assert depth is not None
            layer_px, layer_py, layer_vol_idx = pixel_voxel_corres_given_depth(
                voxel_hash=self._voxel_hash,
                voxel_origin=self._vol_origin,
                voxel_size=self.voxel_size,
                cam_intr=masks_intr,
                cam_pose=cam_pose,
                H=H_m,
                W=W_m,
                depth=depth,
            )

        elif depth_type == "mesh_rc":
            assert o3d is not None, "install open3d 0.17.0 to enable this feature"

            depth = torch.from_numpy(
                self.get_depth_from_mc_mesh(
                    cam_intr=masks_intr,
                    cam_pose=cam_pose,
                    H=H_m,
                    W=W_m,
                )
            ).to(device)
            depth_intr = masks_intr

            layer_px, layer_py, layer_vol_idx = pixel_voxel_corres_given_depth(
                voxel_hash=self._voxel_hash,
                voxel_origin=self._vol_origin,
                voxel_size=self.voxel_size,
                cam_intr=masks_intr,
                cam_pose=cam_pose,
                H=H_m,
                W=W_m,
                depth=depth,
            )

        elif depth_type == "voxel_rc" and o3d is not None:
            depth = torch.from_numpy(
                self.get_depth_from_mc_voxel(
                    cam_intr=masks_intr,
                    cam_pose=cam_pose,
                    H=H_m,
                    W=W_m,
                )
            ).to(device)
            depth_intr = masks_intr
            depth[depth > 0] += self.voxel_size * 0.01

            layer_px, layer_py, layer_vol_idx = pixel_voxel_corres_given_depth(
                voxel_hash=self._voxel_hash,
                voxel_origin=self._vol_origin,
                voxel_size=self.voxel_size,
                cam_intr=masks_intr,
                cam_pose=cam_pose,
                H=H_m,
                W=W_m,
                depth=depth,
            )

        else:
            depth_intr = masks_intr
            layer_px, layer_py, layer_vol_idx, depth = voxel_ray_cast(
                voxel_hash=self._voxel_hash,
                voxel_origin=self._vol_origin,
                voxel_size=self.voxel_size,
                cam_intr=masks_intr,
                cam_pose=cam_pose,
                H=H_m,
                W=W_m,
            )

        # ! STEP 2, map 2d to 3d
        mapped_img, map_2d_3d, self._instance_label_num = instance_2d_map_3d(
            masks=masks,
            voxel_instance=self._instance,
            layer_px=layer_px,
            layer_py=layer_py,
            layer_vol_idx=layer_vol_idx,
            num_3d_instance=self._instance_label_num,
            threshold=threshold,
        )

        # ! STEP 3, fusion according to this mapped_img
        # filter out a layer of voxels that should be updated
        voxel_indices, voxel_px, voxel_py = filter_voxel_from_depth_and_image_and_get_pixels(
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
            feat_intr=masks_intr,
            H_f=H_m,
            W_f=W_m,
            margin=self._margin,
        )

        if voxel_indices is None:
            return map_2d_3d

        # # get all tsdf voxels that is radiated by this view
        old_weight = self._instance_w_sum[voxel_indices]
        old_voxel_labels = self._instance[voxel_indices]
        update_voxel_labels = mapped_img[voxel_py, voxel_px]
        # del pix_x, pix_y

        # update the corresponding weights
        # note that 0 denote unclassified class, in my model, some instance in some view might be unclassified, therefore it may have id=0, for the voxels that are classified as id=0 in current frame, we donot update them.
        identified = update_voxel_labels != 0
        filter_1 = identified & (old_voxel_labels == update_voxel_labels)  # positive results
        filter_2 = identified & (old_voxel_labels != update_voxel_labels) & (old_weight >= obs_weight)  # not fliped
        filter_3 = identified & (old_voxel_labels != update_voxel_labels) & (old_weight < obs_weight)  # fliped

        # case 1 and 2 label unidentified, case 2 and 3 change weight reduce then abs
        self._instance_w_sum[voxel_indices[filter_1]] += obs_weight
        self._instance_w_sum[voxel_indices[filter_2 + filter_3]] = (obs_weight - self._instance_w_sum[voxel_indices[filter_2 + filter_3]]).abs()
        self._instance[voxel_indices[filter_3]] = update_voxel_labels[filter_3]

        return map_2d_3d

    @torch.no_grad()
    def extract_label_on_grid(self, verts: np.ndarray, device=None):
        if device is None:
            device = self.device

        # extract colors, dense features is very expensive, so I choose to use the sparse array
        hash_selected = discrete2hash(discretize_3d(torch.from_numpy(verts.copy()).to(device), voxel_size=self._voxel_size, voxel_origin=self._vol_origin.to(device)))

        hash_new, indices = torch.cat([self._voxel_hash.to(device), hash_selected], dim=0).unique(return_inverse=True, sorted=True)
        idx_old, idx_selected = indices[: self.num_voxel], indices[self.num_voxel :]
        new_n_vox = hash_new.size(0)

        label_merge = torch.zeros(size=(new_n_vox,), dtype=torch.int64, device=device)
        label_merge[idx_old] = self._instance.to(device)

        label_w_sum_merge = self._instance_weight_init + torch.zeros(size=(new_n_vox,), dtype=torch.float32, device=device)
        label_w_sum_merge[idx_old] = self._instance_w_sum.to(device)

        labels = label_merge[idx_selected].cpu().numpy()
        labels_w_sum = label_w_sum_merge[idx_selected].cpu().numpy()

        return labels, labels_w_sum

    def save_instance(self, pth: str):
        torch.save(
            obj={
                "instance_label": self._instance,
                "instance_w_sum": self._instance_w_sum,
                "instance_num": self._instance_label_num,
            },
            f=pth,
        )

    def load_instance(self, pth: str):
        stat_dict = torch.load(pth, map_location=self.device)
        self._instance = stat_dict["instance_label"].to(self.device)
        self._instance_w_sum = stat_dict["instance_w_sum"].to(self.device)
        self._instance_label_num = stat_dict["instance_num"]
