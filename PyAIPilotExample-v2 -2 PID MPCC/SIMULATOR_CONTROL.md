# Simulator restart, countdown, and PnP/EKF/MPCC flight control

The client now follows the restart-aware lifecycle from Version 10:

1. Connect and arm the drone.
2. Hold zero throttle while waiting for a fresh simulator race.
3. Read the simulator race clock and print `3... 2... 1... GO!`.
4. Start metric gate tracking and MPCC flight only at `GO!`.
5. Return to the waiting state when the race schedule clears, a new race is
   scheduled, or the simulator boot clock jumps backward after a restart.

Run normally with:

```powershell
python main.py
```

When `.venv-gpu` is installed, that command automatically relaunches through
the project-local CUDA ONNX Runtime. You can also launch it explicitly with
`.\run_gpu.ps1`. Confirm the startup line says
`Verified ONNX execution provider: CUDAExecutionProvider`.

Press `Ctrl+C` at any time to disarm, stop the receiver threads, and return to
the terminal prompt. The camera receiver has a short timeout, so shutdown also
works when the simulator has stopped sending frames.

To request a simulator reset when the client starts:

```powershell
$env:SIM_RESET_ON_START = "1"
python main.py
```

The four inner YOLO corners are passed to OpenCV PnP with the simulator camera
intrinsics and physical gate opening. The resulting 3D vector is corrected for
the camera's 20 degree pitch. OpenCV's `[right, down, forward]` translation is
reordered before rotation, and every downstream component uses standard body
axes: X forward, Y right, Z down. A six-state EKF tracks relative gate position
and velocity; when the gate vanishes at the crossing, the controller coasts the
state for `GATE_EKF_COAST_S` (1.25 seconds by default). The filter resets when
the simulator reports a new active gate. HIGHRES_IMU samples are retained in a
short timestamped buffer. When delayed PnP arrives, the filter rewinds to the
camera capture time, applies the correction, and replays bias-corrected
accelerometer and gyro samples to the present. Accelerometer bias is learned
while waiting for the race and frozen at `GO`. The displayed raw visual closing
speed remains a direct finite difference of PnP measurements, independent from
the filtered/replayed EKF speed.

CasADi minimizes the finite-horizon contouring objective

```
sum(W_contour * ||e_c||^2 + W_lag * e_l^2
    - W_progress * Delta_s + bounded-control regularization)
```

where the local path tangent points through the estimated hollow gate center.
For unit tangent `t`, predicted drone position `p`, and virtual path progress
`s`, the implementation uses `r(s) = t*s`,
`e_l = t^T*(p - r(s))`, and `e_c = (I - t*t^T)*(p - r(s))`. Thus the lag term
is the along-path error and the contour term is strictly perpendicular to the
line through the gate center.

The first optimized body acceleration is converted to bounded roll, pitch,
yaw, and thrust. Positive body-Y acceleration now maps to positive MAVLink roll
so a gate to the right commands a right bank. Lateral acceleration is capped at
`1.0 m/s^2` and roll at 6 degrees. Within 2 degrees of image center, roll
authority is limited to 1.5 degrees; it increases gradually to the full 6
degrees only at a 15-degree or larger horizontal gate bearing. Roll smoothing
moves 12 percent toward the target per control cycle.

Forward acceleration is capped at `0.8 m/s^2` and braking at `1.0 m/s^2`, but
the final pitch actuator command is temporarily hard-limited to `-0.01` through
`+0.01` degrees in every control mode. This effectively disables forward
acceleration and braking. Normal pitch smoothing is 10 percent per cycle and
overspeed braking uses 25 percent.

The former visual rules remain a fallback when PnP has not yet initialized or
the optimizer is unavailable. Tune the primary weights with
`MPCC_W_CONTOUR`, `MPCC_W_LAG`, and `MPCC_W_PROGRESS`; horizon and step size are
`MPCC_HORIZON` and `MPCC_DT`.

## Safe 5 km/h diagnostic lock

Forward speed and virtual progress are currently locked to 5 km/h (1.389 m/s)
with `MPCC_DEBUG_SPEED_LOCK=1`. CasADi enforces hard velocity-state constraints
at every horizon stage, forward acceleration has matching braking authority,
and an estimate already outside the envelope invokes bounded emergency braking.
The terminal prints the EKF closing speed beside an independent raw PnP
frame-to-frame closing speed. If those diverge, inspect PnP correction frequency
and quality before increasing speed.

After a safe simulator test, disable the temporary lock with
`MPCC_DEBUG_SPEED_LOCK=0`. Then set `MPCC_TARGET_SPEED_MPS`,
`MPCC_MAX_FORWARD_SPEED_MPS`, `MPCC_MAX_LATERAL_SPEED_MPS`, and
`MPCC_MAX_VERTICAL_SPEED_MPS` explicitly. Increase `MPCC_W_PROGRESS` only after
the hard limits and state diagnostics remain trustworthy.

Runtime gate detection requires `models/gate_pose.engine` or
`models/gate_pose.onnx`. Use `export_gate_model.py`; raw `.pt` execution is
disabled unless `ALLOW_PT_RUNTIME=1` is explicitly set for development.

## Terminal status

The aligned terminal block separates gate confidence/range, EKF-versus-visual
speed, detector runtime/ROI latency, camera age, PnP rate, IMU replay count, and
the outgoing command. Set `TERMINAL_STATUS_INTERVAL_S` to change its frequency.
Detailed camera/body axes are hidden by default and can be enabled with
`TERMINAL_VERBOSE_AXES=1`.

Vertical thrust is calculated continuously from MPCC's bounded body-Z
acceleration, which uses both relative gate position and EKF velocity. The
conversion is `T = T_hover / cos(tilt) * (1 - a_down/g)`. Therefore a lower gate
or excessive upward velocity produces sub-hover thrust, while a higher gate or
excessive downward velocity produces more thrust. `INITIAL_HOVER_THRUST=0.28`
is only the gravity feed-forward calibration needed to convert acceleration to
an absolute MAVLink collective; it is not a floor or fixed output. The broad
actuator envelope is `0.25` to `0.48`. The lower bound is the code-authoritative
`ACTIVE_FLIGHT_MIN_THRUST` constant in `controller.py` and is enforced again
immediately before every active-flight MAVLink attitude command. Change that
constant to select a different minimum; the GPU launcher does not override it.

While fresh gate tracking is active, the gate center replaces takeoff altitude
as the vertical reference. This prevents any old altitude-hold correction from
fighting a later gate that is intentionally higher or lower.

Vertical MPCC output is cancelled immediately once the camera measurement is
more than 0.15 seconds old, so an EKF-coasted gate cannot continue commanding a
climb or descent. The terminal shows `HIGH`, `CENTER`, or `LOW`, the requested
body-Z acceleration, calculated thrust, and whether `MPCC-VERTICAL` is active.
The pre-race holding path still sends zero throttle intentionally.

## Optional FPV gate-detection window

The FPV window is disabled by default so GUI rendering cannot steal time from
the detector or flight controller. The terminal still reports all tracking
diagnostics. To enable a throttled preview at 10 FPS:

```powershell
$env:FPV_DISPLAY = "1"
$env:FPV_MAX_FPS = "10"
python main.py
```

The preview displays the live camera image with gate
boxes, confidence, eight YOLO keypoints, inner/outer gate outlines, the gate
center, image-center reticle, and an alignment line. It continues showing the
raw camera feed when a model is unavailable or no gate is detected.

To explicitly run without it:

```powershell
$env:FPV_DISPLAY = "0"
python main.py
```

Runtime temporary files and model caches default to `.runtime` beside the
project on E:. Override that location with `AIPILOT_RUNTIME_DIR` if needed.
