from typing import Callable, Dict

import groundingdino.datasets.transforms as T
import torch
import torchvision
import torchvision.transforms as TS
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image
from ram import inference_ram
from ram.models import ram as get_ram
from ram.models.ram import RAM
from segment_anything import SamPredictor, build_sam_hq
from segment_anything.modeling import Sam

from .base import BaseExtractor, RandomFeatureExtractor


def get_grounding_output(grounding_dino_model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    grounding_dino_model = grounding_dino_model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = grounding_dino_model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = grounding_dino_model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def load_grounding_dino(grounding_dino_config_path, grounding_dino_checkpoint_path, device):
    args = SLConfig.fromfile(grounding_dino_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(grounding_dino_checkpoint_path)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model = model.eval().to(device)
    return model


class GroundedSAMInstanceExtractor(BaseExtractor):
    sam: Sam = None
    ram: RAM = None
    grounding_dino: GroundingDINO = None
    sam_predictor: SamPredictor = None

    def __init__(
        self,
        ram_ckpt: str,
        grounding_dino_config_pth: str,
        grounding_dino_ckpt: str,
        sam_hq_ckpt: str,
        device: str = "cpu",
    ) -> None:
        self.device = device

        self.ram_ckpt = ram_ckpt
        self.grounding_dino_config_pth = grounding_dino_config_pth
        self.grounding_dino_ckpt = grounding_dino_ckpt
        self.sam_hq_ckpt = sam_hq_ckpt

        self.grounding_dino_transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.ram_transform = TS.Compose(
            [
                TS.Resize((384, 384)),
                TS.ToTensor(),
                TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.sam_transform = TS.PILToTensor()

    def load_model(self):
        self.ram = get_ram(pretrained=str(self.ram_ckpt), image_size=384, vit="swin_l").to(self.device).eval()
        self.grounding_dino = load_grounding_dino(
            grounding_dino_config_path=self.grounding_dino_config_pth,
            grounding_dino_checkpoint_path=self.grounding_dino_ckpt,
            device=self.device,
        ).eval()
        self.sam = build_sam_hq(checkpoint=self.sam_hq_ckpt).to(self.device).eval()
        self.sam_predictor = SamPredictor(self.sam)

    @torch.no_grad()
    def extract(self, image: Image) -> Dict:
        if self.sam_predictor is None:
            self.load_model()

        image = image.convert("RGB")

        # step 1: get RAM tages
        res = inference_ram(self.ram_transform(image).unsqueeze(0).to(self.device), self.ram)

        tags = set(["background", "ceiling", "floor"])
        for item in res[0].split("|"):
            if not item.strip().endswith("room"):
                tags.add(item.strip())
        tags = ", ".join(list(tags))

        # step 2:  get grounding dino boxes and others
        image_tensor, _ = self.grounding_dino_transform(image, None)
        boxes_filt, scores, pred_phrases = get_grounding_output(
            self.grounding_dino,
            image_tensor,
            tags,
            box_threshold=0.25,
            text_threshold=0.2,
            device=self.device,
        )

        image_array = self.sam_transform(image).movedim(0, -1).numpy()

        # step 3: get mask by SAM
        self.sam.eval().to(self.device)
        self.sam_predictor.set_image(image_array)
        size = image.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        # nms
        iou_threshold = 0.5
        nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_array.shape[:2]).to(self.device)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        masks = masks.squeeze(1)

        # next filter out wrong artifact masks generated by sam
        arg = torch.argwhere(masks)
        mask_id, py, px = arg[:, 0], arg[:, 1], arg[:, 2]

        down_ratio = 8.0
        H, W = masks.shape[1], masks.shape[2]
        n_masks = masks.shape[0]
        H_new, W_new = int(H / down_ratio), int(W / down_ratio)

        py_new = (py / H * H_new).floor().to(torch.int64)
        px_new = (px / W * W_new).floor().to(torch.int64)

        shrinked_masks = torch.zeros(size=(n_masks, H_new, W_new), dtype=bool, device=masks.device)
        shrinked_masks[mask_id, py_new, px_new] = 1.0

        ratio = (shrinked_masks.count_nonzero(dim=(-1, -2)) / (H_new * W_new)) / (masks.count_nonzero(dim=(-1, -2)) / (H * W))
        valid_mask = torch.argwhere(ratio < 1.4).reshape(-1).cpu().numpy().tolist()

        masks = masks[valid_mask]
        boxes_filt = boxes_filt[valid_mask]
        pred_phrases = [pred_phrases[idx] for idx in valid_mask]

        return {
            "masks": masks,
            "boxes": boxes_filt,
            "phrases": pred_phrases,
        }


class RandomGroundedSAMFeatureExtractor(RandomFeatureExtractor, GroundedSAMInstanceExtractor):
    def __init__(
        self,
        ram_ckpt: str,
        grounding_dino_config_pth: str,
        grounding_dino_ckpt: str,
        sam_hq_ckpt: str,
        feat_dim: int,
        device: str = "cpu",
    ) -> None:
        self.feat_dim = feat_dim

        super().__init__(ram_ckpt, grounding_dino_config_pth, grounding_dino_ckpt, sam_hq_ckpt, device)
