"""Entry point: Train the YOLO segmentation model."""

import traceback

from breast_seg.config import Config
from breast_seg.model import SegmentationModel


def main() -> None:
    try:
        config = Config()
        model = SegmentationModel(config)
        model.train()
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
