# YOLO gate detection

This project uses a YOLO pose model because the supplied annotations contain a
gate bounding box plus eight gate corners. The training and validation folders
are referenced directly; the source data is not copied or modified.

## 1. Install

Use a Python environment supported by the current PyTorch release (Python
3.10-3.12 is the safest choice), then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 2. Validate and train

```powershell
python validate_gate_dataset.py
python train_gate_yolo.py --epochs 100 --batch 16 --device 0
```

Omit `--device 0` to let Ultralytics choose a device, or use `--device cpu`.
Training outputs are written under `runs/gate_pose/train`. The dataset mapping
and keypoint order are in `gate_pose.yaml`.

## 3. Export the accelerated simulator model

Raw PyTorch `.pt` inference is disabled in the final loop. Export the trained
pose model to dynamic ONNX (portable) or TensorRT (NVIDIA GPU):

```powershell
python export_gate_model.py --format onnx
# NVIDIA/TensorRT environment:
python export_gate_model.py --format engine --device 0 --half
python main.py
```

The loop prefers `models/gate_pose.engine`, then `models/gate_pose.onnx`.
Alternatively, set `GATE_MODEL` to an accelerated artifact. `ALLOW_PT_RUNTIME=1`
exists only as an explicit development fallback. Optional runtime settings are
`GATE_CONFIDENCE`, `GATE_IMAGE_SIZE`, and `GATE_DEVICE`.

### NVIDIA CUDA runtime

The RTX path uses a project-local Python environment and CUDA/cuDNN packages on
E:, leaving the base Anaconda environment and C: drive unchanged:

```powershell
.\setup_gpu_runtime.ps1
.\run_gpu.ps1
```

Once installed, `python main.py` also relaunches itself with `.venv-gpu`
automatically. Startup must print
`Verified ONNX execution provider: CUDAExecutionProvider`; CUDA mode treats a
silent CPU fallback as an error. Use `GATE_ONNX_PROVIDER=cpu` only as an
explicit diagnostic fallback. `GATE_CUDA_DEVICE_ID` selects the NVIDIA device.

Benchmark the exact detector path with:

```powershell
.\.venv-gpu\Scripts\python benchmark_gate_runtime.py --force-roi
```

After the first full-frame acquisition, a velocity-predicted square ROI is
expanded around the prior box and inferred at 192 pixels by default. The fixed
square input avoids dynamic cuDNN plan compilation during flight. Boxes and all eight
keypoints are translated back into exact full-frame pixel coordinates before
PnP. Two ROI misses automatically restore full-frame acquisition. Configure
this with `GATE_ROI`, `GATE_ROI_SCALE`, and `GATE_ROI_IMAGE_SIZE`.

The newest result is published to `shared_data["gate_detection"]`. It includes
the frame ID, inference time, all detections, the highest-confidence detection
in `best`, pixel bounding boxes, normalized centers, named keypoints, and a
metric `pose` produced by `cv2.solvePnP()` from the four inner corners.

The default inner opening is 1.5 m x 1.5 m. Override it with `GATE_WIDTH_M` and
`GATE_HEIGHT_M` if a track uses different geometry. PnP uses the supplied
intrinsics `fx=fy=320`, `cx=320`, `cy=180` for 640 x 360 frames (scaled only if
the received image was resized). The returned camera vector is also multiplied
by the configured camera-to-body rotation; `CAMERA_PITCH_DEG` defaults to 20.
Because OpenCV returns `[right, down, forward]`, it is first reordered to
`[forward, right, down]` before applying the supplied rotation matrix. The
resulting body vector consistently uses X-forward, Y-right, Z-down. Both the raw
OpenCV vector and transformed vectors are retained in the pose diagnostics.
Bad corner sets are rejected using their PnP reprojection error.

The camera receiver and flight-control loop are not blocked by model inference.
