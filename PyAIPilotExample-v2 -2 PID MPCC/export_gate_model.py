"""Export trained gate-pose weights for the accelerated simulation runtime."""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent


def default_source():
    candidates = [
        PROJECT_DIR / "models" / "gate_pose.pt",
        PROJECT_DIR.parent / "PyAIPilotExample-v2-1" / "models" / "gate_pose.pt",
        PROJECT_DIR.parent / "PyAIPilotExample-v2-1" / "runs" / "pose" / "runs"
        / "pose" / "gate-finetune-gpu-3" / "weights" / "best.pt",
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--format", choices=("onnx", "engine"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"Trained weights not found: {args.source}")
    output_dir = PROJECT_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    working_source = output_dir / "gate_pose_export_source.pt"
    shutil.copy2(args.source, working_source)

    model = YOLO(str(working_source))
    export_args = {
        "format": args.format,
        "imgsz": args.imgsz,
        "dynamic": args.format == "onnx",
        "simplify": False,
    }
    if args.half:
        export_args["half"] = True
    if args.device:
        export_args["device"] = args.device
    exported = Path(model.export(**export_args))
    destination = output_dir / f"gate_pose.{args.format}"
    if exported.resolve() != destination.resolve():
        shutil.copy2(exported, destination)
        exported.unlink(missing_ok=True)
    working_source.unlink(missing_ok=True)
    print(f"Accelerated gate model: {destination}")


if __name__ == "__main__":
    main()
