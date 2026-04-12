"""
YOLO Segmentation Training Script - Windows Compatible
Model: YOLO11n-seg (ultralytics 8.3)
"""
from ultralytics import YOLO
from breast_seg.config import Config


def main() -> None:
    config = Config()

    # Load YOLO segmentation model (pretrained)
    # Allows flexible training by referencing config.model_name (e.g. yolo11n-seg.pt, yolo26.pt)
    model = YOLO(config.model_name)

    results = model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        patience=config.patience,      # Early stopping trigger
        device=config.device,          # Use GPU block ('0') or 'cpu'
        workers=config.workers,
        project=str(config.runs_dir),
        name="breast_seg_yolo",
        exist_ok=True,
        
        # Data Augmentation (Mammography specific constraints)
        flipud=0.0,                    # Vertical flip strictly disabled for anatomical correctness
        fliplr=0.5,                    # Horizontal flip enabled
        mosaic=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        
        # Optimizer Configuration
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        
        # Segmentation Specific Parameters
        overlap_mask=True,
        mask_ratio=4,
        
        # Verbosity and Callbacks
        verbose=True,
        save=True,
        save_period=10,                # Checkpoint saving frequency
        plots=True,
    )

    print("\n[INFO] === TRAINING COMPLETED ===")
    print(f"[INFO] Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
