"""Asynchronous YOLO gate pose inference for simulator camera frames."""

import ctypes
import logging
import math
import os
import queue
import threading
import time
from pathlib import Path

# CUDA handles the network. Keep PyTorch/OpenMP preprocessing from reserving
# every logical CPU and starving the simulator's render thread between frames.
HOST_PREPROCESS_THREADS = max(1, int(os.environ.get("GATE_HOST_THREADS", "1")))
os.environ.setdefault("OMP_NUM_THREADS", str(HOST_PREPROCESS_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(HOST_PREPROCESS_THREADS))
os.environ.setdefault("KMP_BLOCKTIME", "0")

import numpy as np
import cv2

from gate_pose import GatePoseEstimator


KEYPOINT_NAMES = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "top_left_outer",
    "top_right_outer",
    "bottom_left_outer",
    "bottom_right_outer",
)


class GateDetector:
    """Run inference on the newest available frame without blocking UDP reception."""

    def __init__(
        self,
        weights,
        data,
        confidence=0.35,
        iou=0.45,
        image_size=640,
        device=None,
        gate_width_m=1.5,
        gate_height_m=1.5,
        camera_pitch_deg=20.0,
        roi_enabled=True,
        roi_scale=1.8,
        roi_image_size=192,
    ):
        try:
            from ultralytics import YOLO
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Gate detection requires Ultralytics. Install requirements.txt first."
            ) from exc

        torch.set_num_threads(HOST_PREPROCESS_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # It can only be configured before the first parallel Torch op.
            pass
        cv2.setNumThreads(HOST_PREPROCESS_THREADS)

        self.weights = Path(weights)
        self.data = data
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.runtime = self.weights.suffix.lower().lstrip(".") or "unknown"
        self.device = device
        self.execution_provider = self.runtime.upper()
        self._dll_directory_handles = []
        self._dll_handles = []
        ort_module = None
        original_inference_session = None
        if self.runtime == "onnx":
            ort_module = self._prepare_onnx_runtime()
            original_inference_session = ort_module.InferenceSession
            ort_module.InferenceSession = self._cuda_session_factory(
                ort_module, original_inference_session
            )
        try:
            self.model = YOLO(str(self.weights), task="pose")
        except Exception:
            if ort_module is not None and original_inference_session is not None:
                ort_module.InferenceSession = original_inference_session
            raise
        self.roi_enabled = bool(roi_enabled)
        self.roi_scale = max(1.2, float(roi_scale))
        self.roi_image_size = (
            self.image_size if self.runtime == "engine" else int(roi_image_size)
        )
        self._last_bbox = None
        self._last_bbox_frame_id = None
        self._bbox_velocity = [0.0, 0.0, 0.0, 0.0]
        self._roi_misses = 0
        self.pose_estimator = GatePoseEstimator(
            gate_width_m=gate_width_m,
            gate_height_m=gate_height_m,
            camera_pitch_deg=camera_pitch_deg,
        )
        self.warmup_ms = None
        try:
            if self.runtime in ("onnx", "engine"):
                warmup_started = time.perf_counter()
                warmup_args = {
                    "source": np.zeros((360, 640, 3), dtype=np.uint8),
                    "imgsz": self.image_size,
                    "conf": self.confidence,
                    "verbose": False,
                }
                # CUDA ORT accepts NumPy host input and copies it to the GPU.
                # Ultralytics preprocessing stays on CPU because the installed
                # PyTorch build is intentionally the smaller CPU-only package.
                if self.device and self.execution_provider != "CUDAExecutionProvider":
                    warmup_args["device"] = self.device
                ultralytics_logger = None
                previous_log_level = None
                if self.execution_provider == "CUDAExecutionProvider":
                    from ultralytics.utils import LOGGER

                    ultralytics_logger = LOGGER
                    previous_log_level = LOGGER.level
                    LOGGER.setLevel(logging.WARNING)
                try:
                    self.model.predict(**warmup_args)
                    if (
                        self.runtime == "onnx"
                        and self.roi_enabled
                        and self.roi_image_size != self.image_size
                    ):
                        roi_warmup_args = dict(warmup_args)
                        roi_warmup_args["source"] = np.zeros(
                            (self.roi_image_size, self.roi_image_size, 3),
                            dtype=np.uint8,
                        )
                        roi_warmup_args["imgsz"] = self.roi_image_size
                        self.model.predict(**roi_warmup_args)
                finally:
                    if ultralytics_logger is not None:
                        ultralytics_logger.setLevel(previous_log_level)
                self._verify_active_provider()
                self.warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
        finally:
            if ort_module is not None and original_inference_session is not None:
                ort_module.InferenceSession = original_inference_session
        self._frames = queue.Queue(maxsize=1)
        self._running = True
        # Daemon is a final safeguard: a stuck GPU/runtime call must not keep
        # the terminal alive after the user presses Ctrl+C.
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="gate-detector",
            daemon=True,
        )
        self._thread.start()

    def _prepare_onnx_runtime(self):
        """Load project-local CUDA/cuDNN DLLs before creating an ORT session."""
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is not installed") from exc

        requested = os.environ.get("GATE_ONNX_PROVIDER", "cuda").strip().lower()
        if requested not in {"cuda", "cpu", "auto"}:
            raise ValueError("GATE_ONNX_PROVIDER must be cuda, cpu, or auto")
        available = ort.get_available_providers()
        use_cuda = requested == "cuda" or (
            requested == "auto" and "CUDAExecutionProvider" in available
        )
        if not use_cuda:
            self.execution_provider = "CPUExecutionProvider"
            return ort
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. Launch with the project-local "
                ".venv-gpu Python runtime or set GATE_ONNX_PROVIDER=cpu explicitly."
            )

        if os.name == "nt":
            site_packages = Path(ort.__file__).resolve().parent.parent
            nvidia_root = site_packages / "nvidia"
            bin_directories = sorted(nvidia_root.glob("*/bin"))
            if not bin_directories:
                raise RuntimeError(
                    "Project-local NVIDIA CUDA/cuDNN DLL packages are missing"
                )
            for directory in bin_directories:
                self._dll_directory_handles.append(
                    os.add_dll_directory(str(directory))
                )
            os.environ["PATH"] = os.pathsep.join(
                [*(str(path) for path in bin_directories), os.environ.get("PATH", "")]
            )

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")

        # cuDNN 9.24's loader does not preload these small forwarding DLLs on
        # Windows, although convolution engines load them by name later.
        if os.name == "nt":
            cudnn_bin = Path(ort.__file__).resolve().parent.parent / "nvidia" / "cudnn" / "bin"
            for filename in (
                "cudnn_engines_tensor_ir64_9.dll",
                "cudnn_ext64_9.dll",
                "cudnn_cnn64_9.dll",
            ):
                dll_path = cudnn_bin / filename
                if dll_path.is_file():
                    self._dll_handles.append(ctypes.WinDLL(str(dll_path)))

        self.execution_provider = "CUDAExecutionProvider"
        return ort

    def _cuda_session_factory(self, ort, original_factory):
        """Return an ORT session constructor with an explicit provider policy."""
        execution_provider = self.execution_provider
        cuda_device = int(os.environ.get("GATE_CUDA_DEVICE_ID", "0"))

        def create_session(*args, **kwargs):
            if kwargs.get("sess_options") is None:
                session_options = ort.SessionOptions()
                session_options.log_severity_level = 3
                session_options.intra_op_num_threads = 1
                session_options.inter_op_num_threads = 1
                session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                session_options.add_session_config_entry(
                    "session.intra_op.allow_spinning", "0"
                )
                session_options.add_session_config_entry(
                    "session.inter_op.allow_spinning", "0"
                )
                kwargs["sess_options"] = session_options
            if execution_provider == "CUDAExecutionProvider":
                kwargs["providers"] = [
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": cuda_device,
                            "cudnn_conv_algo_search": "HEURISTIC",
                            "do_copy_in_default_stream": "1",
                        },
                    ),
                    "CPUExecutionProvider",
                ]
            else:
                kwargs["providers"] = ["CPUExecutionProvider"]
            session = original_factory(*args, **kwargs)
            active = session.get_providers()
            if not active or active[0] != execution_provider:
                raise RuntimeError(
                    f"Requested {execution_provider}, but ONNX Runtime activated {active}"
                )
            return session

        return create_session

    def _verify_active_provider(self):
        if self.runtime != "onnx":
            return
        predictor = getattr(self.model, "predictor", None)
        auto_backend = getattr(predictor, "model", None)
        backend = getattr(auto_backend, "backend", auto_backend)
        session = getattr(backend, "session", None)
        if session is None:
            raise RuntimeError("Unable to verify the ONNX Runtime execution provider")
        active = session.get_providers()
        if not active or active[0] != self.execution_provider:
            raise RuntimeError(
                f"ONNX provider fallback detected: requested {self.execution_provider}, "
                f"active providers are {active}"
            )
        print(
            f"Verified ONNX execution provider: {self.execution_provider}",
            flush=True,
        )

    def submit(self, frame_id, image, timestamp_ns=None, received_time_ns=None):
        """Queue a frame, replacing an older frame if inference is still busy."""
        frame = (frame_id, image, timestamp_ns, received_time_ns)
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)

    def stop(self):
        self._running = False
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5.0)

    def _inference_loop(self):
        while self._running:
            try:
                item = self._frames.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                continue

            frame_id, image, timestamp_ns, received_time_ns = item
            started = time.perf_counter()
            try:
                inference_image, roi = self._select_inference_region(
                    image, frame_id
                )
                predict_args = {
                    "source": inference_image,
                    "conf": self.confidence,
                    "iou": self.iou,
                    "imgsz": roi["input_size"],
                    "verbose": False,
                }
                if self.device:
                    predict_args["device"] = self.device
                result = self.model.predict(**predict_args)[0]
                detections = self._serialize_result(
                    result,
                    image.shape[1],
                    image.shape[0],
                    offset_x=roi["x1"],
                    offset_y=roi["y1"],
                )
                best = detections[0] if detections else None
                if best is not None:
                    self._update_roi_tracker(best["bbox"], frame_id)
                    self._roi_misses = 0
                elif roi["used"]:
                    self._roi_misses += 1
                    if self._roi_misses >= 2:
                        self._last_bbox = None
                        self._last_bbox_frame_id = None
                        self._bbox_velocity = [0.0, 0.0, 0.0, 0.0]
                completed = time.time()
                payload = {
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "received_time_ns": received_time_ns,
                    "time": completed,
                    "inference_ms": (time.perf_counter() - started) * 1000.0,
                    "end_to_end_latency_ms": None
                    if received_time_ns is None
                    else (time.time_ns() - int(received_time_ns)) / 1e6,
                    "runtime": self.runtime,
                    "execution_provider": self.execution_provider,
                    "warmup_ms": self.warmup_ms,
                    "roi": roi,
                    "detections": detections,
                    "best": best,
                    "detected": best is not None,
                    "confidence": best["confidence"] if best else 0.0,
                    "offset_x": best["offset_x"] if best else 0.0,
                    "offset_y": best["offset_y"] if best else 0.0,
                    "size_ratio": best["size_ratio"] if best else 0.0,
                    "orientation_deg": best["orientation_deg"] if best else 0.0,
                    "pose": best["pose"] if best else GatePoseEstimator.invalid("no gate detected"),
                    "error": None,
                }
            except Exception as exc:  # keep the camera receiver alive after inference errors
                self._last_bbox = None
                self._last_bbox_frame_id = None
                self._roi_misses = 0
                payload = {
                    "frame_id": frame_id,
                    "timestamp_ns": timestamp_ns,
                    "received_time_ns": received_time_ns,
                    "time": time.time(),
                    "inference_ms": (time.perf_counter() - started) * 1000.0,
                    "end_to_end_latency_ms": None
                    if received_time_ns is None
                    else (time.time_ns() - int(received_time_ns)) / 1e6,
                    "runtime": self.runtime,
                    "execution_provider": self.execution_provider,
                    "warmup_ms": self.warmup_ms,
                    "roi": {"used": False, "input_size": self.image_size},
                    "detections": [],
                    "best": None,
                    "detected": False,
                    "confidence": 0.0,
                    "pose": GatePoseEstimator.invalid("gate inference failed"),
                    "error": str(exc),
                }
            # Assigning a complete dictionary lets readers see a consistent snapshot.
            self.data["gate_detection"] = payload

    def _select_inference_region(self, image, frame_id):
        height, width = image.shape[:2]
        full = {
            "used": False,
            "x1": 0,
            "y1": 0,
            "x2": width,
            "y2": height,
            "input_size": self.image_size,
        }
        if not self.roi_enabled or self._last_bbox is None:
            return image, full

        frame_delta = 1.0
        if self._last_bbox_frame_id is not None:
            frame_delta = max(1.0, float(frame_id - self._last_bbox_frame_id))
        predicted = [
            value + velocity * frame_delta
            for value, velocity in zip(self._last_bbox, self._bbox_velocity)
        ]
        center_x = 0.5 * (predicted[0] + predicted[2])
        center_y = 0.5 * (predicted[1] + predicted[3])
        expansion = self.roi_scale * (1.0 + 0.35 * self._roi_misses)
        roi_side = max(
            96.0,
            (predicted[2] - predicted[0]) * expansion,
            (predicted[3] - predicted[1]) * expansion,
        )
        # A fixed square crop always letterboxes to the same 192x192 tensor.
        # Dynamic rectangular shapes force cuDNN to build new convolution plans
        # during flight and cause visible latency spikes.
        if roi_side > min(width, height):
            return image, full
        roi_side = int(math.ceil(roi_side))
        x1 = int(round(center_x - roi_side * 0.5))
        y1 = int(round(center_y - roi_side * 0.5))
        x1 = min(max(0, x1), width - roi_side)
        y1 = min(max(0, y1), height - roi_side)
        x2 = x1 + roi_side
        y2 = y1 + roi_side
        if x2 - x1 < 64 or y2 - y1 < 64:
            return image, full
        roi = {
            "used": True,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "input_size": self.roi_image_size,
        }
        return image[y1:y2, x1:x2], roi

    def _update_roi_tracker(self, bbox, frame_id):
        current = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        if self._last_bbox is not None and self._last_bbox_frame_id is not None:
            frame_delta = max(1.0, float(frame_id - self._last_bbox_frame_id))
            measured_velocity = [
                (value - previous) / frame_delta
                for value, previous in zip(current, self._last_bbox)
            ]
            self._bbox_velocity = [
                0.65 * old + 0.35 * new
                for old, new in zip(self._bbox_velocity, measured_velocity)
            ]
        self._last_bbox = current
        self._last_bbox_frame_id = frame_id

    def _serialize_result(
        self, result, image_width, image_height, offset_x=0.0, offset_y=0.0
    ):
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        keypoint_rows = None
        if result.keypoints is not None:
            keypoint_rows = result.keypoints.data.cpu().tolist()

        detections = []
        for index, (box, confidence, class_id) in enumerate(zip(boxes, confidences, classes)):
            x1, y1, x2, y2 = box
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y
            keypoints = []
            if keypoint_rows is not None and index < len(keypoint_rows):
                for keypoint_index, values in enumerate(keypoint_rows[index]):
                    x, y = values[:2]
                    x += offset_x
                    y += offset_y
                    score = values[2] if len(values) > 2 else None
                    keypoints.append({
                        "name": KEYPOINT_NAMES[keypoint_index]
                        if keypoint_index < len(KEYPOINT_NAMES) else f"keypoint_{keypoint_index}",
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(score) if score is not None else None,
                    })

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            orientation_deg = 0.0
            if len(keypoints) >= 2:
                left, right = keypoints[0], keypoints[1]
                orientation_deg = math.degrees(math.atan2(
                    right["y"] - left["y"], right["x"] - left["x"]
                ))

            detections.append({
                "class_id": int(class_id),
                "class_name": "gate",
                "confidence": float(confidence),
                "bbox": {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "width": float(x2 - x1), "height": float(y2 - y1),
                },
                "center_normalized": {
                    "x": float(center_x / image_width),
                    "y": float(center_y / image_height),
                },
                "offset_x": float((2.0 * center_x / image_width) - 1.0),
                "offset_y": float((2.0 * center_y / image_height) - 1.0),
                "size_ratio": float((x2 - x1) / image_width),
                "orientation_deg": float(orientation_deg),
                "keypoints": keypoints,
                "pose": self.pose_estimator.estimate(
                    keypoints, image_width=image_width, image_height=image_height
                ),
            })

        detections.sort(key=lambda detection: detection["confidence"], reverse=True)
        return detections
