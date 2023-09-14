from typing import List, Union

import numpy as np
import torch
from skimage import measure
from torch import Tensor

from .utils.tsdf_ops import depth_to_voxel_layer, filter_voxel_from_depth_and_get_tsdf
from .utils.voxel_ops import discrete2hash, discrete2world, hash2discrete, inhomo2homo
from .utils.voxel_to_mesh import get_mesh_from_voxel

try:
    import open3d as o3d
except:
    o3d = None


class ScalableTSDFVolume:
    """Truncated signed distance function (TSDF) to fuse 3D voxels from 2D rgbd images. Actually, most of the feature integration can be found in open3d's voxel block grid (VoxelBlockGrid), however, it is still in development and currently, it cannot manipulate voxels freely (construct from numpy array)."""

    extensive_properties = [
        "_voxel_hash",
        "_tsdf",
        "_tsdf_w_sum",
    ]  # this is a list that grow through time, is list is used in pruning

    # construct tsdf related objects
    _voxel_hash: Tensor  # Note that this array is always sorted!
    _tsdf: Tensor
    _tsdf_w_sum: Tensor

    # these variables are defined after fusion is done
    mc_layer_verts: np.ndarray = None
    mc_layer_faces: np.ndarray = None
    mc_voxel_hash: Tensor = None
    mc_voxel_verts: np.ndarray = None
    mc_voxel_faces: np.ndarray = None

    # these are used when open3d is usable
    mc_layer_raycast: "o3d.t.geometry.RaycastingScene" = None
    mc_voxel_raycast: "o3d.t.geometry.RaycastingScene" = None

    def __init__(
        self,
        voxel_size: float,
        sdf_trunc: float,
        margin: float,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Constructor.

        Args:
            voxel_size (float): The volmue size in meters.
            feat_dim (int): The dimension of the embedding we are going to merge.
            sdf_trunc (float): truncation in meter, a good value is set to 4.0 * voxel_size
            device: torch device.
            margin (float): a value larger than sdf_trun, this is set to preven voxel being filtered out due to large variance of depth
        """

        self.device = device

        # Define voxel volume parameters
        self._voxel_size = voxel_size
        self._vol_origin = Tensor([0, 0, 0]).to(self.device)

        # truncated range
        self._sdf_trunc = sdf_trunc

        # margin
        self._margin = margin

        self.reset()

    def save(self, pth: str):
        torch.save(
            obj={
                "voxel_hash": self._voxel_hash,
                "voxel_size": self._voxel_size,
                "voxel_origin": self._vol_origin,
                "tsdf": self._tsdf,
                "tsdf_w_sum": self._tsdf_w_sum,
            },
            f=pth,
        )

    def load(self, pth: str):
        stat_dict = torch.load(pth, map_location=self.device)
        self._voxel_hash = stat_dict["voxel_hash"].to(self.device)
        self._voxel_size = stat_dict["voxel_size"]
        self._vol_origin = stat_dict["voxel_origin"].to(self.device)
        self._tsdf = stat_dict["tsdf"].to(self.device)
        self._tsdf_w_sum = stat_dict["tsdf_w_sum"].to(self.device)

    @torch.no_grad()
    def reset(self):
        # (N,) for storing hash of voxels
        self._voxel_hash = torch.zeros(size=(0,), dtype=torch.int64, device=self.device)
        # (N,) for storing tsdf volume
        self._tsdf = torch.zeros(size=(0,), dtype=torch.float32, device=self.device)
        # (N,) for storing tsdf weight
        self._tsdf_w_sum = torch.zeros(size=(0,), dtype=torch.float32, device=self.device)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def num_voxel(self):
        return len(self._voxel_hash)

    @torch.no_grad()
    def integrate_tsdf(
        self,
        depth: Tensor,
        depth_intr: Tensor,
        cam_pose: Tensor,
        obs_weight: float = 1.0,
    ):
        """Integrate a only tsdf into TSDF volume.
        Args:
            dense_feat (Tensor): Dense feature prediciton of shape (H, W, feat_dim)
            depth (Tensor): Depth image of shape (H, W)
            depth_intr (Tensor): The camera intrinsic matrix of shape (3, 3)
            cam_pose (Tensor): The camera pose (i.e. extrinsics) of shape (4, 4)
            obs_weight (float): observed weight
        """

        # check input
        assert depth_intr.size() in [(3, 3), (4, 4)], "[!] `depth_intr' should be of shape (3, 3)."
        assert cam_pose.size() == (4, 4), "[!] `cam_pose' should be of shape (4, 4)."

        # get layer of coordinates
        hash_c = depth_to_voxel_layer(
            depth=depth,
            depth_intr=depth_intr,
            cam_pose=cam_pose,
            voxel_origin=self._vol_origin,
            voxel_size=self._voxel_size,
            margin=self._sdf_trunc,
            device=self.device,
            depth_max=depth.max() + self._sdf_trunc,
        )

        if hash_c is None:
            return  # there is no valid points

        # STEP 1: filter voxels
        hash_c, tsdf_upd = filter_voxel_from_depth_and_get_tsdf(
            hash_c=hash_c,
            world_c=inhomo2homo(
                discrete2world(
                    hash2discrete(hash_c),
                    voxel_size=self.voxel_size,
                    voxel_origin=self._vol_origin,
                )
            ),
            cam_pose=cam_pose,
            depth=depth,
            depth_intr=depth_intr,
            sdf_trunc=self._sdf_trunc,
            margin=self._margin,
        )
        del depth

        if hash_c is None:
            return  # there is no valid points

        # STEP 2.0: get new hash list
        # now we update the tsdf part
        voxel_hash, indices = torch.cat([self._voxel_hash, hash_c], dim=0).unique(return_inverse=True, sorted=True)
        new_n_vox = voxel_hash.size(0)
        idx_old, idx_upd = indices[: self.num_voxel], indices[self.num_voxel :]

        # STEP 2.1: update hash and enlarge the data
        if self.num_voxel < new_n_vox:
            self._voxel_hash = voxel_hash  # forge it into the object

            tsdf = torch.ones(size=(new_n_vox,), dtype=torch.float32, device=self.device)
            tsdf[idx_old] = self._tsdf
            tsdf_w_sum = torch.zeros(size=(new_n_vox,), dtype=torch.float32, device=self.device)
            tsdf_w_sum[idx_old] = self._tsdf_w_sum

        else:
            tsdf = self._tsdf
            tsdf_w_sum = self._tsdf_w_sum

        del voxel_hash, indices

        # Step 2.2 forge tsdf
        tsdf_old = tsdf[idx_upd]
        tsdf_w_old = tsdf_w_sum[idx_upd]

        tsdf_w_new = tsdf_w_old + obs_weight
        tsdf[idx_upd] = (tsdf_w_old * tsdf_old + obs_weight * tsdf_upd) / tsdf_w_new
        tsdf_w_sum[idx_upd] = tsdf_w_new

        # forge
        self._tsdf = tsdf
        self._tsdf_w_sum = tsdf_w_sum

        del tsdf, tsdf_w_sum, tsdf_w_old, tsdf_w_new

    @torch.no_grad()
    def get_marching_cube_required_voxels_and_masks(self):
        """
        get a subset of voxels that is required for marching cube calculation
        """
        if self.num_voxel == 0:
            return None, None

        torch.cuda.empty_cache()

        tsdf_sign_sum = self._tsdf.sign().to(int)
        neibor_counts = torch.ones_like(self._voxel_hash)

        x, y, z = np.meshgrid(np.array([0, -1]), np.array([0, -1]), np.array([0, -1]), indexing="ij")  # [0, -1] follows the convention of sklearn marching cube masks
        relative_neighbors = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T[1:].reshape(-1, 1, 3)

        discrete = hash2discrete(self._voxel_hash)
        for neighbor in relative_neighbors:
            dr = torch.from_numpy(neighbor).to(discrete.device)

            neighbor_hash = discrete2hash(discrete + dr)
            neighbor_in_list = torch.isin(neighbor_hash, self._voxel_hash)

            # update neighbor counts
            neibor_counts[neighbor_in_list] += 1

            neighbor_indices = torch.cat(
                [
                    self._voxel_hash,
                    neighbor_hash[neighbor_in_list],
                ],
                dim=0,
            ).unique(return_inverse=True, sorted=True)[
                1
            ][self.num_voxel :]

            # update sign sum
            tsdf_sign_sum[neighbor_in_list] += self._tsdf[neighbor_indices].sign().to(int)

        if (neibor_counts == 8).sum().item() == 0:
            # the current voxels is not enough to run marching cube
            return None, None, None

        mc_mask = (neibor_counts == 8) * ((tsdf_sign_sum < 8) * (tsdf_sign_sum > -8))

        # after we get the neighbor mask (in current list), we extend the list to all its neighbed that is used in mc algorithm

        mc_layer = mc_mask.clone().detach()

        discrete = hash2discrete(self._voxel_hash[mc_mask])
        for neighbor in relative_neighbors:
            dr = torch.from_numpy(neighbor).to(discrete.device)
            neighbor_hash = discrete2hash(discrete + dr)
            neighbor_indices = torch.cat([self._voxel_hash, neighbor_hash], dim=0).unique(return_inverse=True, sorted=True)[1][self.num_voxel :]

            mc_layer[neighbor_indices] = True

        return self._voxel_hash[mc_layer], mc_mask[mc_layer], self._tsdf[mc_layer]

    @torch.no_grad()
    def prune_voxel(self, mild=False):
        """
        During integration, I choose a preservative way to add more voxels. Then I choose to prune some voxels out.
            mild = False: only retains the voxels that is important to marching cube, this is the thin layer of voxels that actualy contributes to marching cube algorithm
                this will reduce the voxel to about 8%
            mild = True: also retains the voxel with tsdf volume that is in range of (-1, 1) and also the other layer of it
        """

        mc_layer_hash = self.get_marching_cube_required_voxels_and_masks()[0]

        retain_mask = torch.isin(self._voxel_hash, mc_layer_hash)

        if mild:
            range_mask = (self._tsdf > -1) * (self._tsdf < 1)
            range_mask_extend = range_mask.clone().detach()

            x, y, z = np.meshgrid(np.array([0, -1, +1]), np.array([0, -1, +1]), np.array([0, -1, +1]), indexing="ij")
            relative_neighbors = np.vstack([x.reshape(-1), y.reshape(-1), z.reshape(-1)]).T[1:].reshape(-1, 1, 3)
            discrete = hash2discrete(self._voxel_hash[range_mask])

            for neighbor in relative_neighbors:
                dr = torch.from_numpy(neighbor).to(discrete.device)

                neighbor_hash = discrete2hash(discrete + dr)
                neighbor_hash = neighbor_hash[(torch.isin(neighbor_hash, self._voxel_hash))]

                indices = torch.cat([self._voxel_hash, neighbor_hash], dim=0).unique(return_inverse=True, sorted=True)[1][self.num_voxel :]
                range_mask_extend[indices] = True

            retain_mask = retain_mask + range_mask_extend

        for attr in self.extensive_properties:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr)[retain_mask])

    @torch.no_grad()
    def extract_mesh(self):
        """extract mesh using marching cube"""
        torch.cuda.empty_cache()

        mc_hash, mc_mask, mc_tsdf = self.get_marching_cube_required_voxels_and_masks()

        # get dense tsdf
        discrete_c = hash2discrete(mc_hash)
        ox, oy, oz = discrete_c[:, 0].min(), discrete_c[:, 1].min(), discrete_c[:, 2].min()  # offset/origin
        nx, ny, nz = discrete_c[:, 0].max() - ox + 1, discrete_c[:, 1].max() - oy + 1, discrete_c[:, 2].max() - oz + 1

        ## create empty array and give values
        dense_tsdf = torch.ones(size=(nx, ny, nz), dtype=torch.float, device=self.device)
        dense_tsdf[discrete_c[:, 0] - ox, discrete_c[:, 1] - oy, discrete_c[:, 2] - oz] = mc_tsdf
        ## also for mask
        masks = torch.zeros(size=(nx, ny, nz), dtype=torch.bool, device=self.device)
        masks[discrete_c[:, 0] - ox, discrete_c[:, 1] - oy, discrete_c[:, 2] - oz] = mc_mask
        # get dense tsdf finish

        verts, faces, _, _ = measure.marching_cubes(dense_tsdf.cpu().numpy(), mask=masks.cpu().numpy(), level=0.0, spacing=(self._voxel_size,) * 3)

        # # convert back
        dense_origin = np.array([[ox.item(), oy.item(), oz.item()]])
        verts = verts + dense_origin * self._voxel_size + self._vol_origin.cpu().numpy().reshape(1, 3)

        torch.cuda.empty_cache()
        return verts, faces

    @torch.no_grad()
    def set_mc_layer_mesh(self):
        """
        This function calculates the meshes of marching cube layer, and the meshes of the voxels/cubes of the marching cube layer
        """

        # marching cube layer
        self.mc_layer_verts, self.mc_layer_faces = self.extract_mesh()

        # marching cube included voxel
        self.mc_voxel_hash = self.get_marching_cube_required_voxels_and_masks()[0]
        _mc_voxel_verts_int, self.mc_voxel_faces = get_mesh_from_voxel(self.mc_voxel_hash)
        self.mc_voxel_verts = (
            discrete2world(
                discrete_coord=_mc_voxel_verts_int,
                voxel_origin=self._vol_origin,
                voxel_size=self.voxel_size,
            )
            - 0.5 * self.voxel_size
        )
        self.mc_voxel_verts = self.mc_voxel_verts.cpu().numpy()
        self.mc_voxel_faces = self.mc_voxel_faces.cpu().numpy()

        if o3d is not None:
            self.mc_layer_raycast = o3d.t.geometry.RaycastingScene()
            mesh = o3d.t.geometry.TriangleMesh(
                self.mc_layer_verts.astype(np.float32),
                self.mc_layer_faces,
            )
            self.mc_layer_raycast.add_triangles(mesh)

            self.mc_voxel_raycast = o3d.t.geometry.RaycastingScene()
            mesh = o3d.t.geometry.TriangleMesh(
                self.mc_voxel_verts.astype(np.float32),
                self.mc_voxel_faces,
            )
            self.mc_voxel_raycast.add_triangles(mesh)

    def get_depth_from_mc_mesh(
        self,
        cam_intr: Tensor,
        cam_pose: Tensor,
        H: int,
        W: int,
    ) -> np.ndarray:
        assert o3d is not None, "install open3d 0.17.0 to enable this feature"

        if self.mc_layer_raycast is None:
            self.set_mc_layer_mesh()

        rays = self.mc_layer_raycast.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(cam_intr[:3, :3].cpu().numpy()),
            extrinsic_matrix=o3d.core.Tensor(torch.linalg.inv(cam_pose).cpu().numpy()),
            width_px=W,
            height_px=H,
        )
        answer = self.mc_layer_raycast.cast_rays(rays)
        depth = answer["t_hit"].numpy()
        depth[np.isinf(depth)] = 0

        return depth

    def get_depth_from_mc_voxel(
        self,
        cam_intr: Tensor,
        cam_pose: Tensor,
        H: int,
        W: int,
    ) -> np.ndarray:
        assert o3d is not None, "install open3d 0.17.0 to enable this feature"

        if self.mc_voxel_raycast is None:
            self.set_mc_layer_mesh()

        rays = self.mc_voxel_raycast.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(cam_intr[:3, :3].cpu().numpy()),
            extrinsic_matrix=o3d.core.Tensor(torch.linalg.inv(cam_pose).cpu().numpy()),
            width_px=W,
            height_px=H,
        )
        answer = self.mc_voxel_raycast.cast_rays(rays)
        depth = answer["t_hit"].numpy()
        depth[np.isinf(depth)] = 0

        return depth
