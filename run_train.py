"""Entry point: Train the YOLO segmentation model."""

from breast_seg.config import Config
from breast_seg.model import SegmentationModel


def main() -> None:
    config = Config()
    model = SegmentationModel(config)
    model.train()


if __name__ == "__main__":
    main()
