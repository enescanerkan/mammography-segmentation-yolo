"""
 Mammography segmentation + karşılaştırma hattı — SOLID OOP orchestrator girişi.
"""
import argparse
import sys
from pathlib import Path

from pipeline.orchestrator import PipelineOrchestrator
from run_train import main as train_main


def main():
    parser = argparse.ArgumentParser(description="Mammography Pipeline (OOP Refactored)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Compare mode
    comp_p = subparsers.add_parser("compare", help="Run the inference and metric evaluations")
    comp_p.add_argument("--limit", type=int, default=0, help="Test set pair limit")
    comp_p.add_argument("--no-viz", action="store_true", help="Disable PNG drawing")
    comp_p.add_argument("--no-dicom-viz", action="store_true", help="Disable DICOM drawing")
    comp_p.add_argument("--out", type=str, default="compare_output", help="Output directory")

    # Train mode
    train_p = subparsers.add_parser("train", help="Run segmentation training")

    args = parser.parse_args()

    if args.mode == "compare":
        root = Path(__file__).resolve().parent
        out_dir = root / args.out
        orchestrator = PipelineOrchestrator(root, out_dir, args.limit, args.no_viz, args.no_dicom_viz)
        orchestrator.run_compare()
    elif args.mode == "train":
        train_main()
    else:
        print(f"Bilinmeyen mod: {args.mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
