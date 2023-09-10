import os
from typing import Dict, List, Set, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from plyfile import PlyData
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms.functional import pil_to_tensor, to_tensor

NYU_ID_TO_COLOR = [
    (0, 0, 0),
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),
    (247, 182, 210),  # desk
    (66, 188, 102),
    (219, 219, 141),  # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14),  # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),  # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),
    (227, 119, 194),  # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  # otherfurn
    (100, 85, 144),
]

SCANNET_DEPTH_SCALE = 1000.0


def class_info(dataset_root: str):
    """return the class name, nyu40 id and scannet raw id"""
    label_df = pd.read_csv(os.path.join(dataset_root, "scannetv2-labels.combined.tsv"), sep="\t")
    label_df.reset_index()

    rawid_to_nyu40id = {0: 0}
    nyu40_id_to_class = {0: "unknown"}

    for _, item in label_df.iterrows():
        rawid_to_nyu40id[item["id"]] = item["nyu40id"]
        nyu40_id_to_class[item["nyu40id"]] = item["nyu40class"]

    return nyu40_id_to_class, rawid_to_nyu40id


class ScanNet(Dataset):
    """
    Dataset object for Scannet. Each __get_item__ returns global information of each scene, including
    """

    def __init__(self, dataset_root: str) -> None:
        # ScanNet control the split of test and train by making a list of scanId, here I only write get item and leave the split to future.

        ## TODO!!! This dataset structure can only give back training data, test data is in another folder

        self.dataset_root = dataset_root
        self.data_root = os.path.join(self.dataset_root, "scans")
        self.scan_id_list = os.listdir(self.data_root)

        # get mapping from raw id to nyu40 id
        label_df = pd.read_csv(os.path.join(dataset_root, "scannetv2-labels.combined.tsv"), sep="\t")
        label_df.reset_index()

        self.rawid_to_nyu40id = {0: 0}
        self.nyu40_id_to_class = {0: "unknown"}

        for iindex, item in label_df.iterrows():
            self.rawid_to_nyu40id[item["id"]] = item["nyu40id"]
            self.nyu40_id_to_class[item["nyu40id"]] = item["nyu40class"]

        self.rawid_to_nyu40id_vectorized = np.vectorize(self.rawid_to_nyu40id.get)

        # get color mapping
        self.nyu40id_to_color = np.array(NYU_ID_TO_COLOR)

    def __len__(self):
        return len(self.scan_id_list)

    def __getitem__(self, index):
        """
        Returns: the dataset of single scan (which returns rgbd and pose matrix), scanId, intrinsic matrix of color and depth, also the resolution of color and depth
        """
        scan_id = self.scan_id_list[index]

        # get intrinsics
        intrinsic_color_path = os.path.join(self.data_root, scan_id, "data/intrinsic/intrinsic_color.txt")
        intrinsic_color_matrix = self.get_intrinsic_matrix(intrinsic_color_path)

        intrinsic_depth_path = os.path.join(self.data_root, scan_id, "data/intrinsic/intrinsic_depth.txt")
        intrinsic_depth_matrix = self.get_intrinsic_matrix(intrinsic_depth_path)

        # get camera information
        meta_file = os.path.join(self.data_root, scan_id, scan_id + ".txt")
        lines = open(meta_file).readlines()
        color_height, color_width, depth_height, depth_width, color_to_depth_extrinsics = None, None, None, None, None
        for line in lines:
            if "colorHeight" in line:
                color_height = int(line.rstrip().strip("colorHeight = "))

            elif "colorWidth" in line:
                color_width = int(line.rstrip().strip("colorWidth = "))

            elif "depthHeight" in line:
                depth_height = int(line.rstrip().strip("depthHeight = "))

            elif "depthWidth" in line:
                depth_width = int(line.rstrip().strip("depthWidth = "))

            elif "colorToDepthExtrinsics" in line:
                color_to_depth_extrinsics = [float(x) for x in line.rstrip().strip("colorToDepthExtrinsics = ").split(" ")]
                color_to_depth_extrinsics = np.array(color_to_depth_extrinsics).reshape((4, 4))

        # get generated vertices, faces, ground truth label, etc
        vh_label_path = os.path.join(self.data_root, scan_id, scan_id + "_vh_clean_2.labels.ply")
        label_data = PlyData.read(vh_label_path)
        verts = np.vstack(
            [
                label_data["vertex"].data["x"],
                label_data["vertex"].data["y"],
                label_data["vertex"].data["z"],
            ]
        ).transpose()
        faces = np.vstack(label_data["face"].data["vertex_indices"])
        gt_labels = np.array(label_data["vertex"].data["label"])
        gt_colors = np.vstack(
            [
                label_data["vertex"].data["red"],
                label_data["vertex"].data["green"],
                label_data["vertex"].data["blue"],
            ]
        ).transpose()

        vh_color_path = os.path.join(self.data_root, scan_id, scan_id + "_vh_clean_2.ply")
        color_data = PlyData.read(vh_color_path)
        rgb_colors = np.vstack(
            [
                color_data["vertex"].data["red"],
                color_data["vertex"].data["green"],
                color_data["vertex"].data["blue"],
            ]
        ).transpose()

        scan_dataset = ScanNetSingleScan(
            data_root=self.data_root,
            scan_id=scan_id,
            color_height=color_height,
            color_width=color_width,
            color_intr=intrinsic_color_matrix,
            depth_height=depth_height,
            depth_width=depth_width,
            depth_intr=intrinsic_depth_matrix,
        )

        return {
            "scan_id": scan_id,
            "scan_dataset": scan_dataset,
            "intrinsic_color_matrix": intrinsic_color_matrix,
            "intrinsic_depth_matrix": intrinsic_depth_matrix,
            "color_height": color_height,
            "color_width": color_width,
            "depth_height": depth_height,
            "depth_width": depth_width,
            "color_to_depth_extrinsics": color_to_depth_extrinsics,
            "vertices": verts,
            "faces": faces,
            "colors": rgb_colors,
            "ground_truth_labels": gt_labels,
            "ground_truth_colors": gt_colors,
            "depth_scale": SCANNET_DEPTH_SCALE,
        }

    def get_intrinsic_matrix(self, intrinsic_file_path: str) -> np.array:
        return np.loadtxt(intrinsic_file_path, delimiter=" ", dtype=np.float32)

    def rawid_to_nyu40id_img_transform(self, img):
        """The label image is in raw id"""
        # first convert potentially PIL image to numpy array
        img_arr = np.array(img)
        img_arr = self.rawid_to_nyu40id_vectorized(img_arr)
        return img_arr


class ScanNetSingleScan(Dataset):
    """
    return pose, rgb img, and depth img
    PS: currently, we don't need 2D label?
    """

    def __init__(
        self,
        data_root: str,
        scan_id: str,
        color_height: int,
        color_width: int,
        color_intr: np.array,
        depth_height: int,
        depth_width: int,
        depth_intr: np.array,
    ):
        """
        scan_id: scene<spaceId>_<scanId>, or scene%04d_%02d, can be obtained from the folder structure
        """
        self.data_root = data_root
        self.scan_id = scan_id

        # get file list
        self.color_img_path = os.path.join(self.data_root, self.scan_id, "data/color")
        self.depth_img_path = os.path.join(self.data_root, self.scan_id, "data/depth")
        self.pose_path = os.path.join(self.data_root, self.scan_id, "data/pose")
        self.label_img_path = os.path.join(self.data_root, self.scan_id, "label-proc")
        self.instance_img_path = os.path.join(self.data_root, self.scan_id, "instance-filt/instance-filt")

        self.color_img_list = os.listdir(self.color_img_path)
        self.color_img_list.sort(key=lambda e: int(e[:-4]))

        self.depth_img_list = os.listdir(self.depth_img_path)
        self.depth_img_list.sort(key=lambda e: int(e[:-4]))

        self.pose_list = os.listdir(self.pose_path)
        self.pose_list.sort(key=lambda e: int(e[:-4]))

        self.label_img_list = os.listdir(self.label_img_path)
        self.label_img_list.sort(key=lambda e: int(e[:-4]))

        self.instance_img_list = os.listdir(self.instance_img_path)
        self.instance_img_list.sort(key=lambda e: int(e[:-4]))

        self.depth_height = depth_height
        self.depth_width = depth_width
        self.depth_intr = depth_intr

        self.color_height = color_height
        self.color_width = color_width
        self.color_intr = color_intr

        self.resize_to_depth = transforms.Resize(size=(depth_height, depth_width))

        # NOTE: I would better leave the image as its original and do transformation at processing time.
        # # transform
        # self.transform = transforms.Compose([
        #     ## Might need resize to feed into clip model
        #     ## Might also need different transform for depth and rgb data
        #     transforms.ToTensor(),  # convert PIL to tensor
        #     # transforms.ConvertImageDtype(dtype=torch.float32),  # convert to torch.float32
        # ])

    def __len__(self):
        return len(self.color_img_list)

    def get_pose_matrix(self, pose_file_path: str) -> np.array:
        return np.loadtxt(pose_file_path, delimiter=" ", dtype=np.float32)

    def get_item_path(self, index):
        """
        Temporary use because open3d denies the conversion from PIL image, and the detailed reason is in C++.
        So I need to prove this is working in Python for only once.
        """
        # id check
        color_id = self.color_img_list[index].replace(".jpg", "")
        depth_id = self.depth_img_list[index].replace(".png", "")
        pose_id = self.pose_list[index].replace(".txt", "")
        label_id = self.label_img_list[index].replace(".png", "")
        instance_id = self.instance_img_list[index].replace(".png", "")

        if color_id != depth_id or color_id != pose_id or color_id != label_id or color_id != instance_id:
            raise Exception("ID Error: Color Img, Depth Img and Pose Id inconsistent!")

        # get path
        color_file_name = os.path.join(self.color_img_path, self.color_img_list[index])
        depth_file_name = os.path.join(self.depth_img_path, self.depth_img_list[index])
        pose_file_name = os.path.join(self.pose_path, self.pose_list[index])
        label_file_name = os.path.join(self.label_img_path, self.label_img_list[index])
        instance_file_name = os.path.join(self.instance_img_path, self.instance_img_list[index])

        return {
            "color_file_name": color_file_name,
            "depth_file_name": depth_file_name,
            "pose_file_name": pose_file_name,
            "label_file_name": label_file_name,
            "instance_file_name": instance_file_name,
        }

    def __getitem__(self, index):
        # get path
        paths = self.get_item_path(index)

        # read file
        depth_img = Image.open(paths["depth_file_name"])
        color_img = Image.open(paths["color_file_name"])
        pose_matrix = self.get_pose_matrix(paths["pose_file_name"])
        label_img = Image.open(paths["label_file_name"])
        instance_img = Image.open(paths["instance_file_name"])

        return {
            "color_img": color_img,
            "depth_img": depth_img,
            "pose_matrix": pose_matrix,
            "label_img": label_img,
            "instance_img": instance_img,
            "color_intr": self.color_intr,
            "depth_intr": self.depth_intr,
        }

    def get_torch_tensor(
        self,
        index,
        device: Union[torch.device, str] = "cpu",
        keys: Union[List[str], Set[str]] = [
            "depth",
            "color",
            "pose",
            "label",
            "instance",
            "color_intr",
            "depth_intr",
        ],
    ):
        """
        Given index and device, return the wanted keys's torch tensor on wanted device.
            depth: returns depth in float32 and in unit of meter
            color: returns rgb in format of HxWx3 in range of [0, 1]
            pose:
        For image in mode 'I', pil_to_tensor and to_tensor both return integer value
        For image in mode 'L' and 'RGB', pil_to_tensor return integer, while to_tensor returns [0, 1] float
        """
        files_dict = self.get_item_path(index)
        results = {}

        if "depth" in keys:
            results["depth"] = pil_to_tensor(Image.open(files_dict["depth_file_name"])).squeeze(0).to(device, torch.float32) / SCANNET_DEPTH_SCALE

        if "color" in keys:
            results["color"] = to_tensor(Image.open(files_dict["color_file_name"])).squeeze(0).to(device).movedim(0, -1)

        if "pose" in keys:
            results["pose"] = torch.from_numpy(self.get_pose_matrix(files_dict["pose_file_name"])).to(device)

        if "label" in keys:
            results["label"] = pil_to_tensor(Image.open(files_dict["label_file_name"])).squeeze(0).to(device)

        if "instance" in keys:
            results["instance"] = pil_to_tensor(Image.open(files_dict["instance_file_name"])).squeeze(0).to(device)

        if "color_intr" in keys:
            results["color_intr"] = torch.from_numpy(self.color_intr).to(device)

        if "depth_intr" in keys:
            results["depth_intr"] = torch.from_numpy(self.depth_intr).to(device)

        return results
