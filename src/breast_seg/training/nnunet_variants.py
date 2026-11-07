"""nnU-Net v2 trainer variants for mammography.

This file is *installed* into ``nnunetv2/training/nnUNetTrainer/variants/`` by
:func:`breast_seg.training.nnunet.install_trainer_variants` (nnU-Net only discovers trainers inside its own
package); it is never imported by ``breast_seg`` itself.

Deviations from the default trainer, all anatomical or schedule-related:

1. **No vertical mirroring.** On an MLO view the pectoral muscle is always at the top and on a CC view the breast
   always points away from the chest wall; an upside-down mammogram does not exist. Horizontal mirroring stays
   (left and right breasts both occur), matching the YOLO / SegFormer recipes (``fliplr`` on, ``flipud`` off).
   Inference mirroring is restricted the same way.
2. **Fixed epoch count.** ``NUM_EPOCHS`` epochs of nnU-Net's fixed 250 iterations; nnU-Net has no early stopping,
   it keeps ``checkpoint_best`` alongside ``checkpoint_final``.

The full-image 640x640 patch is not set here but in the plans (see ``patch_size`` in the training config).
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerMammo(nnUNetTrainer):
    NUM_EPOCHS = 300

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.NUM_EPOCHS

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rot, dummy_2d, patch, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = (1,)  # 2D: axis 0 = up-down, axis 1 = left-right
        self.inference_allowed_mirroring_axes = mirror_axes
        return rot, dummy_2d, patch, mirror_axes


class nnUNetTrainerMammo_1000epochs(nnUNetTrainerMammo):
    NUM_EPOCHS = 1000
