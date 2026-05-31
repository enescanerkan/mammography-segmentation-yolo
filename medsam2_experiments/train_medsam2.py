"""
MedSAM2 image fine-tune: runs bowang-lab `finetune_sam2_img.py` with README defaults.

From `medsam2_experiments` (conda env `medsam2`, CUDA PyTorch):

  $env:KMP_DUPLICATE_LIB_OK = "TRUE"
  python prepare_medsam2_data.py --use-project-splits
  python train_medsam2.py

Optional: `python train_medsam2.py --prepare` runs prepare first.
Override: `--epochs`, `--batch-size`, `--lr`, `--bbox-shift`, `--device`, `--resume`, `--dry-run`.
SAM2 @ 1024² is VRAM-heavy: default batch is 2 for ~8GB GPUs; use `--batch-size 1` if OOM persists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch MedSAM2 finetune_sam2_img.py with sane defaults.")
    parser.add_argument("--prepare", action="store_true", help="Run prepare_medsam2_data.py --use-project-splits first.")
    parser.add_argument("--task-name", type=str, default="MammoBNP_v1")
    parser.add_argument("--epochs", type=int, default=100, help="Matches README fine-tune example.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-step batch (1024² encoder). Use 1 on tight 8GB; 8+ needs large VRAM.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="README default; try 6e-5 if unstable.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bbox-shift", type=int, default=10, help="Box jitter (256-space) around GT bbox.")
    parser.add_argument("--num-workers", type=int, default=0, help="Windows: keep 0. Linux: try 4.")
    parser.add_argument("--device", type=str, default=None, help="Default: cuda:0 if CUDA else cpu.")
    parser.add_argument("--resume", type=str, default=None, help="Path to medsam_model_latest.pth or best.")
    parser.add_argument("--medsam-repo", type=Path, default=None, help="Override MEDSAM2_REPO / config default.")
    parser.add_argument("--train-npy", type=Path, default=None, help="Override train folder (imgs+gts).")
    parser.add_argument("--work-dir", type=Path, default=None, help="Override work_dir for checkpoints.")
    parser.add_argument("--pretrain", type=Path, default=None, help="Override SAM2 base weights (tiny .pt).")
    parser.add_argument("--model-cfg", type=str, default=None, help="Override sam2 yaml name.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit.")
    args = parser.parse_args()

    exp_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(exp_root))
    from config import (
        EXP_ROOT,
        MEDSAM2_REPO,
        NPY_OUT_ROOT,
        SAM2_BASE_WEIGHTS,
        SAM2_MODEL_CFG,
    )

    repo = Path(args.medsam_repo or MEDSAM2_REPO).resolve()
    finetune = repo / "finetune_sam2_img.py"
    if not finetune.is_file():
        raise SystemExit(
            f"finetune_sam2_img.py not found under:\n  {repo}\n"
            "Clone MedSAM2 branch, e.g. vendor/MedSAM, or set MEDSAM2_REPO / --medsam-repo."
        )

    train_npy = (args.train_npy or (NPY_OUT_ROOT / "train")).resolve()
    gts = train_npy / "gts"
    imgs = train_npy / "imgs"
    if not gts.is_dir() or not imgs.is_dir():
        raise SystemExit(f"Need {gts} and {imgs} with paired .npy files.\nRun: python prepare_medsam2_data.py --use-project-splits")
    n = len(list(gts.glob("*.npy")))
    if n == 0:
        raise SystemExit(
            f"No training .npy under {gts}.\nRun: python prepare_medsam2_data.py --use-project-splits"
        )

    work_dir = (args.work_dir or (EXP_ROOT / "work_dir")).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    pretrain = Path(args.pretrain or SAM2_BASE_WEIGHTS).resolve()
    if not pretrain.is_file():
        raise SystemExit(
            f"Missing SAM2 base weights:\n  {pretrain}\n"
            "Download sam2_hiera_tiny.pt into medsam2_experiments/checkpoints/ (see README)."
        )

    model_cfg = args.model_cfg or SAM2_MODEL_CFG
    device = args.device
    if device is None:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.prepare:
        prep = [sys.executable, str(exp_root / "prepare_medsam2_data.py"), "--use-project-splits"]
        print("Running:", " ".join(prep))
        subprocess.run(prep, cwd=str(exp_root), check=True)

    cmd = [
        sys.executable,
        str(finetune),
        "-i",
        str(train_npy),
        "-task_name",
        args.task_name,
        "-work_dir",
        str(work_dir),
        "-batch_size",
        str(args.batch_size),
        "-num_epochs",
        str(args.epochs),
        "-lr",
        str(args.lr),
        "-weight_decay",
        str(args.weight_decay),
        "-bbox_shift",
        str(args.bbox_shift),
        "-num_workers",
        str(args.num_workers),
        "-pretrain_model_path",
        str(pretrain),
        "-model_cfg",
        model_cfg,
        "-device",
        device,
    ]
    if args.resume:
        cmd.extend(["-resume", str(Path(args.resume).resolve())])

    env = os.environ.copy()
    pp = str(repo)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = pp + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = pp
    # Reduces fragmentation on CUDA (PyTorch docs); harmless if unset by user.
    if str(device).startswith("cuda") and not env.get("PYTORCH_CUDA_ALLOC_CONF"):
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Training samples (gts/*.npy): {n}")
    print(f"cwd for finetune: {repo}")
    print("Command:\n  " + " ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, cwd=str(repo), env=env, check=True)


if __name__ == "__main__":
    main()
