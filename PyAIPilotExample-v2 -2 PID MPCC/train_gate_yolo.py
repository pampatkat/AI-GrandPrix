"""Train an Ultralytics YOLO pose model on the supplied gate dataset."""

import argparse
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "gate_pose.yaml")
    parser.add_argument("--model", default="yolo11n-pose.pt",
                        help="Base pose checkpoint or an existing .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16,
                        help="Use -1 for automatic batch sizing")
    parser.add_argument("--device", default=None,
                        help="For example: 0, 0,1, cpu, or mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=PROJECT_DIR / "runs" / "gate_pose")
    parser.add_argument("--name", default="train")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {args.data}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    train_args = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project.resolve()),
        "name": args.name,
        "patience": args.patience,
        "exist_ok": True,
    }
    if args.device:
        train_args["device"] = args.device
    if args.resume:
        train_args["resume"] = True

    results = model.train(**train_args)
    save_dir = Path(results.save_dir)
    print(f"Training complete. Best weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
