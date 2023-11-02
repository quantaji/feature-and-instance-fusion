from typing import Dict, List

import clip
import groundingdino.datasets.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as TS
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image
from ram.models import ram as get_ram
from ram.models.ram import RAM
from ram.utils.openset_utils import article, multiple_templates, processed_name
from segment_anything import SamPredictor, build_sam_hq
from segment_anything.modeling import Sam
from skimage.measure import label as count_components
from skimage.morphology import binary_dilation
from transformers import AutoTokenizer

from .base import BaseExtractor, RandomFeatureExtractor
from .scannet_labels import CLASS_LABELS_20, CLASS_LABELS_200, VALID_CLASS_IDS_20, VALID_CLASS_IDS_200


def build_openset_label_embedding(
    categories,
    device: str = "cuda:0",
):
    """
    modifiied from ram.utils.build_openset_label_embedding for better device and download control
    """
    model, _ = clip.load(
        "ViT-B/16",
        device=device,
    )
    templates = multiple_templates

    with torch.no_grad():
        openset_label_embedding = []
        for category in categories:
            texts = [template.format(processed_name(category, rm_dot=True), article=article(category)) for template in templates]
            texts = ["This is " + text if text.startswith("a") or text.startswith("the") else text for text in texts]
            texts = clip.tokenize(texts).to(device=device)  # tokenize

            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            openset_label_embedding.append(text_embedding)

        openset_label_embedding = torch.stack(openset_label_embedding, dim=1)

    openset_label_embedding = openset_label_embedding.t()
    return openset_label_embedding


def get_ram_output(
    model: RAM,
    image: torch.Tensor,
):
    """
    Modified from ram.models.ram.RAM.generate_tag_openset. This function gives string output, I have to convert it back to label ids. This is unnecessary.
    """
    label_embed = torch.nn.functional.relu(model.wordvec_proj(model.label_embed))

    image_embeds = model.image_proj(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

    # recognized image tags using image-tag recogntiion decoder
    image_spatial_embeds = image_embeds[:, 1:, :]

    bs = image_spatial_embeds.shape[0]
    label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode="tagging",
    )

    logits = model.fc(tagging_embed[0]).squeeze(-1)

    targets = torch.where(torch.sigmoid(logits) > model.class_threshold.to(image.device), torch.tensor(1.0).to(image.device), torch.zeros(model.num_class).to(image.device))

    tag = targets.cpu().numpy()
    tag[:, model.delete_tag_index] = 0
    indices = np.argwhere(tag == 1)[:, 1]

    return indices


def tags_to_caption(tags: List[str]) -> str:
    """
    Given a set of nouns, convert it into Grounding DINO compatible captions.
    """
    caption = ", ".join(tags)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    return caption


def get_phrases_id_from_logit(
    logit: torch.Tensor,
    tags: List[str],
    tokenizer: AutoTokenizer,
    text_threshold: float = 0.2,
):
    """
    There is a one to many mapping from a box to possible tag's tokenzier, therefore, the original method is not guaranteed to output a single category. Here we match a bounding box with the most probable tag.
    Input:
        logit: N_box x max_token_length
    """

    # 101 [CLS]
    # 102 [SEP]
    # 1010 ','
    # 1012 '.'

    tokenized = tokenizer(tags_to_caption(tags))
    token_ids = np.array(tokenized["input_ids"])

    token_splits = np.split(np.arange(token_ids.shape[0]), np.argwhere(token_ids == 1010).reshape(-1))

    effective_logit = logit[:, : token_ids.shape[0]]

    logit_mask = (effective_logit > text_threshold) * torch.from_numpy(~np.isin(token_ids, [101, 102, 1010, 1012])).to(logit.device).reshape(1, -1)

    effective_logit = effective_logit * logit_mask

    tag_mask = torch.stack(
        [logit_mask[:, single_split].any(dim=1) for single_split in token_splits],
        dim=1,
    )

    tag_logit = torch.stack(
        [effective_logit[:, single_split].max(dim=1)[0] for single_split in token_splits],
        dim=1,
    )

    tag_logit[~tag_mask] = -torch.inf

    pred_tag_idx = tag_logit.argmax(dim=1)
    pred_tag_mask = ~torch.isinf(tag_logit.max(dim=1)[0])
    # pred_tag = [tags[i] for i in pred_tag_idx]
    pred_score = tag_logit.max(dim=1)[0]

    return pred_tag_idx, pred_tag_mask, pred_score


def get_grounding_output(
    grounding_dino_model,
    image,
    tags,
    box_threshold,
    text_threshold,
    device="cpu",
):
    """
    Get bounding box, not the final output.
    """
    caption = tags_to_caption(tags=tags)
    grounding_dino_model = grounding_dino_model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = grounding_dino_model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output by logits
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # tag prediction
    pred_tag_idx, valid_tag_filter, score = get_phrases_id_from_logit(
        logit=logits_filt,
        tags=tags,
        tokenizer=grounding_dino_model.tokenizer,
        text_threshold=text_threshold,
    )

    ## filter out failure decoding
    score = score[valid_tag_filter]
    boxes_filt = boxes_filt[valid_tag_filter]
    pred_tag_idx = pred_tag_idx[valid_tag_filter]

    return boxes_filt, score, pred_tag_idx


def load_grounding_dino(grounding_dino_config_path, grounding_dino_checkpoint_path, device):
    args = SLConfig.fromfile(grounding_dino_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(grounding_dino_checkpoint_path)
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model = model.eval().to(device)
    return model


class GroundedSAMInstanceExtractor(BaseExtractor):
    name = "grounded_sam"

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
        tag_set: str,
        device: str = "cpu",
    ) -> None:
        self.device = device

        self.ram_ckpt = ram_ckpt
        self.grounding_dino_config_pth = grounding_dino_config_pth
        self.grounding_dino_ckpt = grounding_dino_ckpt
        self.sam_hq_ckpt = sam_hq_ckpt

        assert tag_set in ["built_in", "scannet_20", "scannet_200"]
        self.tag_set = tag_set

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
        # load ram
        self.ram = get_ram(pretrained=str(self.ram_ckpt), image_size=384, vit="swin_l").to(self.device).eval()

        # modify tag
        if self.tag_set != "built_in":
            if self.tag_set == "scannet_20":
                categories = list(CLASS_LABELS_20)
            elif self.tag_set == "scannet_200":
                categories = list(CLASS_LABELS_200)
            else:
                raise NotImplementedError

            category_embedding = build_openset_label_embedding(categories=categories, device=self.device)
            self.ram.tag_list = np.array(categories)
            self.ram.label_embed = nn.Parameter(category_embedding.float().to(self.device))
            self.ram.num_class = len(categories)
            self.ram.class_threshold = torch.ones(self.ram.num_class) * 0.5
            self.ram.eval().to(self.device)

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

        # step 1: get RAM tags
        ram_results = get_ram_output(
            model=self.ram,
            image=self.ram_transform(image).unsqueeze(0).to(self.device),
        )
        tags = self.ram.tag_list[ram_results].tolist()

        # step 2:  get grounding dino boxes and others
        boxes_filt, scores, relative_label = get_grounding_output(
            grounding_dino_model=self.grounding_dino,
            image=self.grounding_dino_transform(image, None)[0],
            tags=tags,
            box_threshold=0.25,
            text_threshold=0.2,
            device=self.device,
        )
        # shifted semantic label
        labels = ram_results[relative_label].reshape(-1)

        # convert relative to absolute box
        size = image.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        # nms filtering
        nms_filter = torchvision.ops.nms(boxes_filt, scores, 0.5).numpy().tolist()
        boxes_filt = boxes_filt[nms_filter]
        scores_filt = scores[nms_filter]
        label_filt = labels[nms_filter]

        # step 3: get mask by SAM
        image_array_sam = self.sam_transform(image).movedim(0, -1).numpy()

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_array_sam.shape[:2]).to(self.device)

        self.sam_predictor.set_image(image_array_sam)
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        masks = masks.squeeze(1)  # n_mask, H, W

        # filter out defect segmentation
        # this gives the connected component of each masts,
        # if there are too many
        # this is probably a defect mask
        num_components = np.array([count_components(binary_dilation(mask.cpu().numpy()), return_num=True)[1] for mask in masks])
        sam_defect_filter = num_components < 60

        boxes_filt = boxes_filt[sam_defect_filter]
        scores_filt = scores_filt[sam_defect_filter]
        label_filt = label_filt[sam_defect_filter]
        masks_filt = masks[sam_defect_filter]

        return {
            "masks": masks_filt,
            "boxes": boxes_filt,
            "labels": label_filt,
            "phrases": [self.ram.tag_list[label] for label in label_filt],
        }


class RandomGroundedSAMFeatureExtractor(RandomFeatureExtractor, GroundedSAMInstanceExtractor):
    def __init__(
        self,
        ram_ckpt: str,
        grounding_dino_config_pth: str,
        grounding_dino_ckpt: str,
        sam_hq_ckpt: str,
        tag_set: str,
        feat_dim: int,
        device: str = "cpu",
    ) -> None:
        self.feat_dim = feat_dim

        super().__init__(
            ram_ckpt=ram_ckpt,
            grounding_dino_config_pth=grounding_dino_config_pth,
            grounding_dino_ckpt=grounding_dino_ckpt,
            sam_hq_ckpt=sam_hq_ckpt,
            tag_set=tag_set,
            device=device,
        )
