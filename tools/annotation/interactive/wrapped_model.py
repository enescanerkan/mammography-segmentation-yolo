"""
MedSAM2 wrapper matching bowang-lab/MedSAM MedSAM2 branch (box prompts as corner points).

Used for fine-tuned checkpoint inference. Duplicated here so this repo does not
import private symbols from MedSAM at install time; keep in sync with upstream
`infer_medsam2_flare22.py` / `finetune_sam2_img.py` if their forward changes.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MedSAM2(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.sam2_model = model
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image: torch.Tensor, box: torch.Tensor):
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
                concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, *_ = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        return low_res_masks_logits

    def _image_encoder(self, input_image: torch.Tensor):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


@torch.no_grad()
def medsam2_segment_from_box(
    medsam_model: MedSAM2,
    img_rgb_uint8,
    box_xyxy,
    device: torch.device,
    sam2_transforms,
):
    """Return (binary_mask HxW uint8, prob map HxW float32) in original image space."""
    import numpy as np

    H, W = img_rgb_uint8.shape[:2]
    img_1024_tensor = sam2_transforms(img_rgb_uint8.copy())[None, ...].to(device)
    box_1024 = box_xyxy.astype(np.float32) / np.array([W, H, W, H], dtype=np.float32) * 1024.0
    box_np = np.asarray(box_1024.reshape(1, 4), dtype=np.float32)
    low_res_masks_logits = medsam_model(img_1024_tensor, box_np)
    low_res_pred = torch.sigmoid(low_res_masks_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    prob = low_res_pred.squeeze().cpu().numpy().astype(np.float32)
    seg = (prob > 0.5).astype(np.uint8)
    return seg, prob
