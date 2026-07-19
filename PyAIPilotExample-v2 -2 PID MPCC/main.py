"""Restart-aware PnP/EKF/MPCC gate flight client."""

import os
import sys
import time
from pathlib import Path


# These must be set before setup.py imports OpenCV, NumPy, or Ultralytics.
HOST_THREADS = max(1, int(os.environ.get("GATE_HOST_THREADS", "1")))
os.environ["OMP_NUM_THREADS"] = str(HOST_THREADS)
os.environ["MKL_NUM_THREADS"] = str(HOST_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(HOST_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(HOST_THREADS)
os.environ["KMP_BLOCKTIME"] = "0"

# Keep this project's temporary/model caches off a space-constrained C: drive.
# This runs before importing Ultralytics/OpenCV through setup.py.
RUNTIME_ROOT = Path(
    os.environ.get(
        "AIPILOT_RUNTIME_DIR",
        Path(__file__).resolve().parent / ".runtime",
    )
)
for runtime_subdir in ("temp", "cache", "torch", "ultralytics"):
    (RUNTIME_ROOT / runtime_subdir).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(RUNTIME_ROOT / "temp")
os.environ["TMP"] = str(RUNTIME_ROOT / "temp")
os.environ["XDG_CACHE_HOME"] = str(RUNTIME_ROOT / "cache")
os.environ["TORCH_HOME"] = str(RUNTIME_ROOT / "torch")
os.environ["YOLO_CONFIG_DIR"] = str(RUNTIME_ROOT / "ultralytics")


def _relaunch_with_project_gpu_runtime():
    """Make `python main.py` transparently use the E:-local CUDA runtime."""
    if os.environ.get("AIPILOT_AUTO_GPU_RUNTIME", "1") == "0":
        return
    gpu_python = Path(__file__).resolve().parent / ".venv-gpu" / "Scripts" / "python.exe"
    if not gpu_python.is_file():
        return
    if Path(sys.executable).resolve() == gpu_python.resolve():
        return
    os.execv(
        str(gpu_python),
        [str(gpu_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


if __name__ == "__main__":
    _relaunch_with_project_gpu_runtime()

from setup import setup_components


SIM_SERVER_UDP_IP = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550


def main():
    shared_data = {}
    components = setup_components(
        shared_data,
        int(time.time() * 1000),
        SIM_SERVER_UDP_IP,
        SIM_SERVER_UDP_PORT,
    )
    controller = components["controller"]
    try:
        if os.environ.get("SIM_RESET_ON_START", "0") == "1":
            print("Requesting simulator reset...", flush=True)
            controller.send_sim_reset_command()
            time.sleep(0.5)
        print("Client ready. Press Ctrl+C at any time to stop and return to the terminal.", flush=True)
        while True:
            print("Arming drone...", flush=True)
            controller.arm()
            controller.wait_until_armed()
            controller.run_countdown()
            print("PnP/EKF gate tracking and MPCC control active.", flush=True)
            controller.fly_until_reset()
            print("Simulator reset/new race detected; preparing again.", flush=True)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping and returning to the terminal...", flush=True)
    finally:
        controller.hover()
        controller.disarm()
        vision_thread = components["vision_rx"].get_thread_for_join()
        fpv_display = components.get("fpv_display")
        if fpv_display is not None:
            fpv_display.stop()
        detector = components.get("gate_detector")
        if detector is not None:
            detector.stop()
        components["ts_loop"].get_thread_for_join().join(timeout=1.0)
        components["mavlink_rx"].get_thread_for_join().join(timeout=1.0)
        vision_thread.join(timeout=1.0)
        print("Client exited.", flush=True)


if __name__ == "__main__":
    main()
