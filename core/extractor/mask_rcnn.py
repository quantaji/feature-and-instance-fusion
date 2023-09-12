from typing import Dict

import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image

from .base import BaseExtractor, RandomFeatureExtractor


class MaskRCNNMaskExtractor(BaseExtractor):
    cfg: CfgNode = None
    predictor: DefaultPredictor = None

    def __init__(
        self,
        mask_rcnn_ckpt: str,
        device: str,
    ) -> None:
        self.mask_rcnn_ckpt = mask_rcnn_ckpt
        self.device = device

    def load_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        if self.mask_rcnn_ckpt is not None:
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 40
            cfg.MODEL.WEIGHTS = self.mask_rcnn_ckpt
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.DEVICE = self.device

        self.predictor = DefaultPredictor(cfg)

    @torch.no_grad()
    def extract(self, image: Image) -> Dict:
        image = image.convert("RGB")
        img_np = np.array(image)[:, :, ::-1]

        outputs = self.predictor(img_np)

        return {
            "boxes": outputs["instances"].pred_boxes.tensor,
            "classes": outputs["instances"].pred_classes,
            "masks": outputs["instances"].pred_masks,
            "score": outputs["instances"].scores,
        }
