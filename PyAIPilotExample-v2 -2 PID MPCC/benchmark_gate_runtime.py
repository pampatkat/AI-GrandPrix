"""Benchmark the configured gate detector and report its active provider."""

import argparse
import os
import statistics
import time

# Apply the same process-start limits as main.py before importing NumPy/OpenCV.
HOST_THREADS = max(1, int(os.environ.get("GATE_HOST_THREADS", "1")))
os.environ["OMP_NUM_THREADS"] = str(HOST_THREADS)
os.environ["MKL_NUM_THREADS"] = str(HOST_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(HOST_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(HOST_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"

import cv2
import numpy as np

from gate_detector import GateDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional 640x360 validation image")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--force-roi",
        action="store_true",
        help="Benchmark a synthetic tracked-gate ROI instead of full-frame search",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=0.0,
        help="Throttle submissions to a camera-like rate; zero runs flat out",
    )
    args = parser.parse_args()

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Unable to read {args.image}")
    else:
        image = np.zeros((360, 640, 3), dtype=np.uint8)

    data = {}
    detector = GateDetector("models/gate_pose.onnx", data, roi_image_size=192)
    timings = []
    try:
        benchmark_started = time.perf_counter()
        for frame_id in range(1, args.iterations + 1):
            if args.rate_hz > 0.0:
                target_time = benchmark_started + (frame_id - 1) / args.rate_hz
                remaining = target_time - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
            if args.force_roi:
                detector._last_bbox = [270.0, 110.0, 370.0, 250.0]
                detector._last_bbox_frame_id = frame_id - 1
                detector._roi_misses = 0
            detector.submit(
                frame_id,
                image,
                timestamp_ns=time.time_ns(),
                received_time_ns=time.time_ns(),
            )
            deadline = time.time() + 10.0
            while time.time() < deadline:
                result = data.get("gate_detection") or {}
                if result.get("frame_id") == frame_id:
                    if result.get("error"):
                        raise RuntimeError(result["error"])
                    timings.append(float(result["inference_ms"]))
                    break
                time.sleep(0.001)
            else:
                raise TimeoutError(f"Detector timed out on frame {frame_id}")
    finally:
        detector.stop()

    print(f"Provider : {detector.execution_provider}")
    print(f"Warmup   : {detector.warmup_ms:.1f} ms")
    print(f"Mean     : {statistics.mean(timings):.2f} ms")
    print(f"Median   : {statistics.median(timings):.2f} ms")
    print(f"Maximum  : {max(timings):.2f} ms")


if __name__ == "__main__":
    main()
