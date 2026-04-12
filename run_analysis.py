"""Entry point: Run MLO/CC geometric analysis (Pectoral Line + PNL + CC Depth).

- MLO views: pectoral muscle boundary line + perpendicular PNL from nipple
- CC views: horizontal depth line from nipple to breast tissue edge
"""

from breast_seg.config import Config
from breast_seg.model import SegmentationModel
from breast_seg.analyzer import MLOAnalyzer
from breast_seg.visualizer import ResultVisualizer


def main() -> None:
    config = Config()
    config.ensure_output_dirs()
    output_dir = config.analysis_output_dir

    # 1. Predict
    print("[1/3] Running segmentation inference...")
    model = SegmentationModel(config, weights=config.weights_path)
    yolo_results = model.predict(
        source=config.test_images_dir,
        save=False,
        save_txt=False,
        show_labels=False,
        show_conf=False,
    )

    # 2. Analyze
    print("[2/3] Analyzing predictions...")
    analyzer = MLOAnalyzer(config)
    analysis_results = analyzer.analyze_predictions(yolo_results)

    # 3. Visualize & Save
    print("[3/3] Generating visualizations...")
    visualizer = ResultVisualizer(config)

    mlo_count = 0
    cc_count = 0
    pnl_count = 0
    cc_depth_count = 0

    for result in analysis_results:
        saved_path = visualizer.draw_and_save(result, output_dir)
        name = saved_path.stem

        if result.is_mlo:
            mlo_count += 1
            if result.pnl:
                pnl_count += 1
                print(f"  [MLO] PNL={result.pnl.distance_px:.0f}px  {name}")
            else:
                print(f"  [MLO] no nipple  {name}")
        else:
            cc_count += 1
            if result.cc_depth:
                cc_depth_count += 1
                side = result.cc_depth.breast_side.upper()
                print(f"  [CC]  Depth={result.cc_depth.distance_px:.0f}px ({side})  {name}")
            else:
                print(f"  [CC]  no measurement  {name}")

    # Summary
    total = len(analysis_results)
    print(f"\n{'='*55}")
    print(f"  Total images:         {total}")
    print(f"  MLO views:            {mlo_count}  (PNL computed: {pnl_count})")
    print(f"  CC views:             {cc_count}  (Depth computed: {cc_depth_count})")
    print(f"  Results saved to:     {output_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
