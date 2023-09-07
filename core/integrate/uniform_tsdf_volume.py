import torch
import numpy as np
from typing import Union
from skimage import measure


class UniformTSDFVolume:
    """Truncated signed distance function (TSDF) to fuse 3D voxels from 2D rgbd images."""

    def __init__(
        self,
        vol_bnds: Union[np.ndarray, list],
        voxel_size: float,
        feat_dim: int,
        sdf_trunc: Union[float, None] = None,
        device: torch.device = torch.device("cpu"),
        include_var: bool = False,
    ) -> None:
        """Constructor.

        Args:
            vol_bnds (ndarray or list): An ndarray of shape (3, 2). Specifies the volume bounds (x,y,z's min/max) in meters.
            voxel_size (float): The volmue size in meters.
            feat_dim (int): The dimension of the embedding we are going to merge.
            sdf_trunc (float): truncation in meter, default is set to 5*voxel_size
            device: torch device.
            include_var (bool): whether the variance is calculated for feature vector.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        self.device = device

        # Define voxel volume parameters
        self._voxel_size = voxel_size
        self._vol_bnds = torch.from_numpy(vol_bnds).float().to(self.device)

        # how many voxels in xyz dim
        self._vol_dim = torch.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).long()
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + (self._vol_dim * self._voxel_size)  # adjust to discretized
        self._vol_origin = self._vol_bnds[:, 0]  # at the corner
        self._num_voxels = torch.prod(self._vol_dim).item()
        if sdf_trunc is None:
            self._sdf_trunc = 5 * self._voxel_size

        # feature parameter
        self._feat_dim = feat_dim
        self._total_dim = self._vol_dim.tolist() + [self._feat_dim]

        # calculation param
        self._include_var = include_var

        # Get voxel grid coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self._vol_dim[0]),
            torch.arange(0, self._vol_dim[1]),
            torch.arange(0, self._vol_dim[2]),
            indexing="ij",
        )
        self._vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)  # (num_voxel, 3) denoting their voxel coordinate

        # Convert voxel coordinates to world coordinates
        self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
        self._world_c = torch.cat(
            [self._world_c, torch.ones(len(self._world_c), 1, device=self.device)],
            dim=1,
        )  # (num_voxel, 4), in perspective coordinate

        # construct
        self.reset()

    def reset(self):
        # averaged signed distance function
        self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
        self._tsdf_w_sum_vol = torch.zeros(*self._vol_dim).to(self.device)  # sum of tsdf's weight
        self._feat_w_sum_vol = torch.zeros(*self._vol_dim).to(self.device)  # sum of feature's weight
        self._feat_vol = torch.zeros(*self._total_dim).to(self.device)
        if self._include_var:
            # sum of the squares of feature's weight
            self._feat_w2_sum_vol = torch.zeros(*self._vol_dim).to(self.device)
            # sum of the squares of feature vector
            self._feat_square_vol = torch.zeros(*self._total_dim).to(self.device)

    def feat_var(self):
        """Calculate the variance of feature
        sample_reliability_variance = S / (w_sum - w_sum2 / w_sum)
        # NOTE & TODO: this can calculation can be query specific, not global
        """
        # when there is only 1 measurement, this will give infinity, it must be cliped
        var = (1e-16 + self._feat_square_vol) / (1e-8 + self._feat_w_sum_vol - (1e-32 + self._feat_w2_sum_vol) / (1e-8 + self._feat_w_sum_vol)).unsqueeze(-1)
        return torch.clamp(var, max=1e8)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def feat_dim(self):
        return self._feat_dim

    @torch.no_grad()
    def integrate(
        self,
        dense_feat: torch.Tensor,
        depth_img: torch.Tensor,
        cam_intr: torch.Tensor,
        cam_pose: torch.Tensor,
        obs_weight: float = 1.0,
        depth_scale: float = 1000.0,
    ):
        """Integrate a dense feature prediction into TSDF volume.
        Args:
            dense_feat (torch.Tensor): Dense feature prediciton of shape (H, W, feat_dim)
            depth_img (torch.Tensor): Depth image of shape (H, W)
            cam_intr (torch.Tensor): The camera intrinsic matrix of shape (3, 3)
            cam_pose (torch.Tensor): The camera pose (i.e. extrinsics) of shape (4, 4)
            obs_weight (float): observed weight
            depth_scale (float): The scale of depth map, usually the dpeth is given in millimeter, so the scale is by default 1000.0.
        """

        # check input
        assert cam_intr.size() in [(3, 3), (4, 4)], "[!] `cam_intr' should be of shape (3, 3)."
        assert cam_pose.size() == (4, 4), "[!] `cam_pose' should be of shape (4, 4)."
        assert dense_feat.size()[:2] == depth_img.size()[:2], "[!] `dense_feat' and `depth_img' should have same size (H, W)."
        assert dense_feat.size(2) == self.feat_dim, "[!] `dense_feat's third dimension should be equal to TSDF Volume's feature dimension."

        H, W = depth_img.size()

        # scale transform
        depth_img = depth_img.to(torch.get_default_dtype()) / depth_scale  # convert it to float

        # Convert world coordinates to camera coordinates
        world2cam = torch.inverse(cam_pose)
        cam_c = torch.matmul(world2cam, self._world_c.transpose(1, 0)).transpose(1, 0).float()

        # Convert camera coordinates to pixel coordinates
        fx, fy = cam_intr[0, 0], cam_intr[1, 1]
        cx, cy = cam_intr[0, 2], cam_intr[1, 2]
        pix_z = cam_c[:, 2]
        pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
        pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

        # Filter 1: remove pixel outside the frame
        valid_pix = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H) & (pix_z > 0)
        valid_vox_x = self._vox_coords[valid_pix, 0]
        valid_vox_y = self._vox_coords[valid_pix, 1]
        valid_vox_z = self._vox_coords[valid_pix, 2]
        valid_pix_y = pix_y[valid_pix]
        valid_pix_x = pix_x[valid_pix]
        depth_val = depth_img[pix_y[valid_pix], pix_x[valid_pix]]

        # Filter 2: remove pixel outside truncated value
        depth_diff = depth_val - pix_z[valid_pix]
        valid_pts = (depth_val > 0) & (depth_diff >= -self._sdf_trunc)
        valid_vox_x = valid_vox_x[valid_pts]
        valid_vox_y = valid_vox_y[valid_pts]
        valid_vox_z = valid_vox_z[valid_pts]
        valid_pix_y = valid_pix_y[valid_pts]
        valid_pix_x = valid_pix_x[valid_pts]

        dist = torch.clamp(depth_diff / self._sdf_trunc, max=1)
        valid_dist = dist[valid_pts]

        # stop if there is nothing
        if 0 in valid_dist.size():
            return

        # integration for tsdf
        tsdf_w_old = self._tsdf_w_sum_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_old = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        # update
        tsdf_w_new = tsdf_w_old + obs_weight
        self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (tsdf_w_old * tsdf_old + obs_weight * valid_dist) / tsdf_w_new
        self._tsdf_w_sum_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_w_new

        # NOTE:
        # In traditional TSDF, we always add 1 to the empty space, so that the sdf space will likely to be 1.
        # but in feature fusion, we do not need empty space to be added again and again.

        # Filter 3: remove pixel outside positive side of truncated value
        valid_pts_2 = depth_diff[valid_pts] <= self._sdf_trunc
        valid_vox_x = valid_vox_x[valid_pts_2]
        valid_vox_y = valid_vox_y[valid_pts_2]
        valid_vox_z = valid_vox_z[valid_pts_2]
        valid_pix_y = valid_pix_y[valid_pts_2]
        valid_pix_x = valid_pix_x[valid_pts_2]

        # integrate feature
        feat_w_old = self._feat_w_sum_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        feat_old = self._feat_vol[valid_vox_x, valid_vox_y, valid_vox_z]

        feat_obs = dense_feat[valid_pix_y, valid_pix_x]

        feat_w_new = feat_w_old + obs_weight

        feat_new = feat_old + obs_weight * (feat_obs - feat_old) / feat_w_new.unsqueeze(-1)

        self._feat_vol[valid_vox_x, valid_vox_y, valid_vox_z] = feat_new
        self._feat_w_sum_vol[valid_vox_x, valid_vox_y, valid_vox_z] = feat_w_new

        if self._include_var:
            feat_w2_old = self._feat_w2_sum_vol[valid_vox_x, valid_vox_y, valid_vox_z]
            feat_square_old = self._feat_square_vol[valid_vox_x, valid_vox_y, valid_vox_z]

            feat_w2_new = feat_w2_old + obs_weight**2
            feat_square_new = feat_square_old + obs_weight * (feat_obs - feat_old) * (feat_obs - feat_new)

            self._feat_w2_sum_vol[valid_vox_x, valid_vox_y, valid_vox_z] = feat_w2_new
            self._feat_square_vol[valid_vox_x, valid_vox_y, valid_vox_z] = feat_square_new

        # NOTE: The online calculation for weighted sum's variance is
        # for x, w in data_weight_pairs:
        #     w_sum = w_sum + w
        #     w_sum2 = w_sum2 + w**2
        #     mean_old = mean
        #     mean = mean_old + (w / w_sum) * (x - mean_old)
        #     S = S + w * (x - mean_old) * (x - mean)

    def extract_point_cloud(self):
        """Extract a point cloud from the voxel volume"""
        tsdf_vol = self._tsdf_vol.cpu().numpy()
        feat_vol = self._feat_vol.cpu().numpy()
        vol_origin = self._vol_origin.cpu().numpy()

        # Marching cubes
        verts = measure.marching_cubes(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + vol_origin

        # Get vertex features, if it is rgb, then it is in scale [0, 1]
        feat_vals = feat_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]

        return verts, feat_vals
