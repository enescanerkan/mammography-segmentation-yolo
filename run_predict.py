"""Entry point: Run segmentation prediction on test images."""

from breast_seg.config import Config
from breast_seg.model import SegmentationModel


def main() -> None:
    config = Config()
    config.ensure_output_dirs()

    model = SegmentationModel(config, weights=config.weights_path)
    results = model.predict(source=config.test_images_dir)

    print(f"\n[INFO] Prediction complete. {len(results)} images processed.")
    print(f"[INFO] Results saved to: {config.predictions_dir}")


if __name__ == "__main__":
    main()
