import os
from pathlib import Path

from pymavlink import mavutil
from timesync import TimeSync
from vision_rx import VisionRX
from mavlink_rx import MAVLinkRX
from controller import Controller
from gate_detector import GateDetector
from fpv_display import FPVDisplay
from imu_buffer import IMUHistory

def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    shared_data.setdefault("imu_buffer", IMUHistory(
        max_age_s=float(os.environ.get("IMU_BUFFER_SECONDS", "2.0"))
    ))
    # -------------------------------
    # Mavlink Connection
    # -------------------------------
    # Start a connection listening on a UDP port
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port,))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # -------------------------------
    # Setup Mavlink msg receiver
    # -------------------------------
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # -------------------------------
    # Timesync request Loop
    # -------------------------------
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # -------------------------------
    # Connect Vision receiver
    # -------------------------------
    project_dir = Path(__file__).resolve().parent
    configured_model = os.environ.get("GATE_MODEL")
    allow_pt_runtime = os.environ.get("ALLOW_PT_RUNTIME", "0") == "1"
    model_candidates = [
        Path(configured_model) if configured_model else None,
        project_dir / "models" / "gate_pose.engine",
        project_dir / "models" / "gate_pose.onnx",
        project_dir.parent / "PyAIPilotExample-v2-1" / "models" / "gate_pose.engine",
        project_dir.parent / "PyAIPilotExample-v2-1" / "models" / "gate_pose.onnx",
    ]
    if allow_pt_runtime:
        model_candidates.extend([
            project_dir / "models" / "gate_pose.pt",
            project_dir.parent / "PyAIPilotExample-v2-1" / "models" / "gate_pose.pt",
            project_dir.parent / "PyAIPilotExample-v2-1" / "runs" / "pose" / "runs"
            / "pose" / "gate-finetune-gpu-3" / "weights" / "best.pt",
        ])
    weights = next(
        (candidate for candidate in model_candidates if candidate and candidate.is_file()),
        None,
    )
    gate_detector = None
    if weights is not None and weights.suffix.lower() == ".pt" and not allow_pt_runtime:
        weights = None
    if weights is not None and weights.is_file():
        print(f"Loading gate detector: {weights}", flush=True)
        try:
            gate_detector = GateDetector(
                weights,
                shared_data,
                confidence=float(os.environ.get("GATE_CONFIDENCE", "0.25")),
                image_size=int(os.environ.get("GATE_IMAGE_SIZE", "640")),
                device=os.environ.get("GATE_DEVICE") or None,
                gate_width_m=float(os.environ.get("GATE_WIDTH_M", "1.5")),
                gate_height_m=float(os.environ.get("GATE_HEIGHT_M", "1.5")),
                camera_pitch_deg=float(os.environ.get("CAMERA_PITCH_DEG", "20.0")),
                roi_enabled=os.environ.get("GATE_ROI", "1") != "0",
                roi_scale=float(os.environ.get("GATE_ROI_SCALE", "1.8")),
                roi_image_size=int(os.environ.get("GATE_ROI_IMAGE_SIZE", "192")),
            )
            shared_data["gate_detector_status"] = "ready"
            shared_data["gate_model_path"] = str(weights)
            shared_data["gate_model_runtime"] = weights.suffix.lower().lstrip(".")
            shared_data["gate_execution_provider"] = gate_detector.execution_provider
            print(
                f"Gate inference provider: {gate_detector.execution_provider}",
                flush=True,
            )
        except Exception as exc:
            shared_data["gate_detector_status"] = "runtime_unavailable"
            shared_data["gate_detector_error"] = str(exc)
            print(f"Gate detector disabled: {exc}", flush=True)
    else:
        shared_data["gate_detector_status"] = "accelerated_model_missing"
        print(
            "Gate detector disabled: export models/gate_pose.onnx or .engine "
            "(raw .pt runtime is disabled).",
            flush=True,
        )
    fpv_display = None
    if os.environ.get("FPV_DISPLAY", "0") != "0":
        preview_fps = float(os.environ.get("FPV_MAX_FPS", "10"))
        print(f"Opening FPV preview at up to {preview_fps:g} FPS...", flush=True)
        fpv_display = FPVDisplay(shared_data, max_fps=preview_fps)
    vision_rx = VisionRX(shared_data, gate_detector, fpv_display)

    # -------------------------------
    # Main control loop
    # -------------------------------
    controller = Controller(sim_conn, shared_data, system_boot_ms)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'gate_detector': gate_detector,
        'fpv_display': fpv_display,
    }
