# COMPREHENSIVE NAVIGATION.PY SPECIFICATIONS

## Overview & Purpose

`navigation.py` implements FPS-style keyboard and mouse control for drone flight. The module translates user input (WASD for horizontal movement, E/C for camera-relative altitude changes, mouse for camera rotation) into NED-frame velocity commands sent to the flight controller at 250 Hz. ALL movement (horizontal and vertical) is relative to the camera's local axes (i.e., affected by camera yaw, pitch, and roll). The camera and vehicle body share the same origin; the camera is mounted tilted 20° upward relative to the body (camera pitch = body pitch + 20°). Mouse-driven camera motion and autonomous velocity inputs are planned as future integrations and should be supported by the attitude composition described below.

---

## 1. REQUIREMENTS

### Functional Requirements

| Requirement | Input | Output | Notes |
|---|---|---|---|
| **Forward/Backward** | W key (forward) / S key (backward) | Velocity in camera-forward direction at 2.0 m/s | Regardless of drone heading |
| **Left/Right Strafe** | A key (left) / D key (right) | Velocity perpendicular to camera-forward (±90°) at 2.0 m/s | Perpendicular to forward direction |
| **Altitude Control** | E key (up) / C key (down) | Velocity along camera-local vertical at 2.0 m/s (camera-up/down) | Camera-relative: affected by camera pitch and roll |
| **Diagonal Movement** | W+A, W+D, S+A, S+D | Vector sum of individual directions at 2.0 m/s per axis | E.g., W+A = forward + left = 45° angle at (2.0, -2.0) NED |
| **Conflict Resolution** | Conflicting keys (W+S or A+D) | Conflicting axes cancel to 0 | W+S → vx=0; A+D → vy=0; both → hover |
| **No Input** | No keys pressed | vx=0, vy=0, vz=0 | Drone hovers |
| **Continuous Polling** | Key events | State checked every control cycle (every 4ms) | Handles key press/release at sub-frame speed |
| **Heading-Aware Movement** | Any WASD input + drone yaw | Movement direction rotates with drone heading from MAVLink | Forward always = camera direction |

### Non-Functional Requirements

| Requirement | Specification |
|---|---|
| **Update Frequency** | Called every control cycle: 250 Hz (every 4 ms) |
| **Velocity Frame** | NED (North-East-Down): vx=north, vy=east, vz=down(positive=DOWN) |
| **Speed Magnitude** | 2.0 m/s for all directions (lateral and vertical) |
| **Response Latency** | <4ms (next control cycle at latest) |
| **Velocity Application** | Instant change (no acceleration ramping) |
| **Camera Attitude Source** | `armed_controller.data['attitude']['roll']`, `['pitch']`, `['yaw']` (radians) |
| **Coordinate Transform** | Camera-relative movement (WASD + E/C) → NED frame using full attitude rotation (roll, pitch, yaw) |

---

## 2. COORDINATE FRAMES & MATHEMATICAL FOUNDATION

### NED Frame (Used by Controller API)
- **North (vx)**: Positive north, negative south
- **East (vy)**: Positive east, negative west  
- **Down (vz)**: Positive down, negative up (note: inverted from typical +Z up convention)
- **Yaw (ψ)**: Drone heading in radians; 0 = north, +π/2 = east, +π or -π = south, -π/2 = west

### Camera-Relative Frame (User Input WASD + E/C)
- Camera axes (camera/body frame):
    - `x` (forward): positive in the direction the camera is pointing
    - `y` (right): positive to the camera's right (pressing `D` yields positive `y`)
    - `z` (down): positive down relative to the camera (consistent with NED's down sign)
- **Forward (W)**: positive `x` in camera frame
- **Left (A)**: negative `y` in camera frame
- **Backward (S)**: negative `x` in camera frame
- **Right (D)**: positive `y` in camera frame
- **Up/Down (E/C)**: up = negative `z` in camera frame (E → negative `z`), down = positive `z` (C → positive `z`)

### Transformation: Camera-Relative → NED (Full 3D)
Camera-relative velocities are expressed in the camera/body frame as a 3-vector `v_cam = [vx_cam, vy_cam, vz_cam]` where `vx_cam` is forward, `vy_cam` is right, and `vz_cam` is down (positive down). To convert `v_cam` to NED/inertial coordinates `v_ned`, apply the rotation defined by the camera/world attitude (roll φ, pitch θ, yaw ψ):

```
R = R_z(psi) * R_y(theta) * R_x(phi)

# where
R_x(phi) = [[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]]
R_y(theta) = [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
R_z(psi) = [[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]]

v_ned = R * v_cam
```

This reduces to the 2D yaw-only rotation when `phi = 0` and `theta = 0` (level camera).

**Example (level camera, phi=0, theta=0, psi=π/2):**
- User presses W (forward): `v_cam = [2, 0, 0]` → `v_ned = [0, 2, 0]` (moves East)

**Example (pitched camera):**
- If camera is pitched down (θ > 0), pressing E (camera-up) will produce an upward component in NED plus forward/back components depending on pitch and roll.

### Altitude (camera-relative `vz`)
- In camera frame: `vz_cam` is positive down. Key mappings (camera frame):
    - **E key**: `vz_cam = -2.0` (camera-up)
    - **C key**: `vz_cam = +2.0` (camera-down)
    - **No input**: `vz_cam = 0`

After building the full camera-relative velocity vector `[vx_cam, vy_cam, vz_cam]`, transform it to NED using the full attitude rotation `R(roll,pitch,yaw)` described above, and send the resulting `v_ned` to the controller API.

---

## 3. FUNCTION SPECIFICATIONS

### Primary Function: `handle_user_input(armed_controller)`

**Purpose:** Main entry point; called every control cycle (250 Hz).

**Signature:**
```python
def handle_user_input(armed_controller: Controller) -> None
```

**Inputs:**
- `armed_controller` (Controller object):
  - `.sim_conn`: MAVLink connection object (passed to flight control API)
  - `.system_boot_ms`: Boot timestamp in milliseconds (passed to flight control API)
  - `.data`: Shared data dict containing:
        - `.data['attitude']`: Dict representing the vehicle body attitude with keys `roll`, `pitch`, `yaw` (floats, radians). The camera attitude is computed from the body attitude by applying a fixed camera mount offset (camera pitch +20°) and any optional mouse-driven camera deltas.
    - Other state keys (optional for this function)

**Outputs:**
- None (side effect: calls `controller.update_position_flight_control()`)

**Behavior:**
1. Poll all relevant keys (W, A, S, D, E, C)
2. Resolve conflicts (W+S → both cancel, A+D → both cancel)
3. Build velocity vector from remaining keys
4. Retrieve body attitude from `armed_controller.data['attitude']` (use roll, pitch, yaw)
     - Compute `camera_attitude` by applying the fixed camera mount offset (pitch +20°) and any optional mouse-driven deltas:
         `camera_attitude.roll = body_roll + mouse_roll_delta` (if provided)
         `camera_attitude.pitch = body_pitch + CAMERA_PITCH_OFFSET + mouse_pitch_delta`
         `camera_attitude.yaw = body_yaw + mouse_yaw_delta`
5. Transform camera-relative velocity to NED frame using `camera_attitude` full rotation (roll, pitch, yaw)
6. Call `controller.update_position_flight_control(armed_controller.sim_conn, armed_controller.system_boot_ms, velocity_dict)`
7. Return (no output)

**Error Handling:**
- If `armed_controller.data['attitude']` is missing or incomplete: log warning, assume roll=0,pitch=0,yaw=0
- If `armed_controller.sim_conn` is None: silently return (MAVLink unavailable)
- If `armed_controller.system_boot_ms` is missing: raise ValueError (critical timing info)

---

### Helper Function: `_get_pressed_keys() -> Dict[str, bool]`

**Purpose:** Query keyboard state for all navigation keys.

**Signature:**
```python
def _get_pressed_keys() -> Dict[str, bool]
```

**Outputs:**
```python
{
    'w': bool,  # True if W key currently pressed
    'a': bool,
    's': bool,
    'd': bool,
    'e': bool,
    'c': bool
}
```

**Behavior:**
- Call `keyboard.is_pressed()` for each key
- Return dict with boolean values
- **Note:** Must not raise exceptions; if keyboard unavailable, return all False

---

### Helper Function: `_resolve_conflicts(keys: Dict[str, bool]) -> Dict[str, bool]`

**Purpose:** Cancel conflicting keypresses (W+S, A+D).

**Signature:**
```python
def _resolve_conflicts(keys: Dict[str, bool]) -> Dict[str, bool]
```

**Input:**
```python
{
    'w': bool,
    'a': bool,
    's': bool,
    'd': bool,
    'e': bool,
    'c': bool
}
```

**Output:** Same structure with conflicts resolved:
```python
{
    'w': bool,  # False if both W and S pressed
    'a': bool,  # False if both A and D pressed
    's': bool,  # False if both W and S pressed
    'd': bool,  # False if both A and D pressed
    'e': bool,  # (no conflict resolution needed)
    'c': bool   # (no conflict resolution needed)
}
```

**Conflict Resolution Rules:**
1. If `keys['w']` and `keys['s']` both True: set both to False
2. If `keys['a']` and `keys['d']` both True: set both to False
3. If `keys['e']` and `keys['c']` both True: set both to False (altitude conflict)
4. Return modified dict

**Example:**
- Input: `{'w': True, 's': True, 'a': False, 'd': True, 'e': False, 'c': False}`
- After resolution: `{'w': False, 's': False, 'a': False, 'd': True, 'e': False, 'c': False}`

---

### Helper Function: `_calculate_velocity_camera_frame(keys: Dict[str, bool]) -> Dict[str, float]`

**Purpose:** Convert resolved key presses to camera-relative velocity components.

**Signature:**
```python
def _calculate_velocity_camera_frame(keys: Dict[str, bool]) -> Dict[str, float]
```

**Input:** Resolved keys dict (from `_resolve_conflicts()`)

**Output:**
```python
{
    'vx': float,  # Camera-relative forward/backward [m/s]
    'vy': float,  # Camera-relative left/right [m/s]
    'vz': float   # Altitude in camera frame [m/s]; camera convention: positive=down
}
```

**Calculation:**
- `vx = (2.0 if keys['w'] else 0.0) + (-2.0 if keys['s'] else 0.0)`
- `vy = (-2.0 if keys['a'] else 0.0) + (2.0 if keys['d'] else 0.0)`
- `vz = (2.0 if keys['c'] else 0.0) + (-2.0 if keys['e'] else 0.0)`

**Examples:**
- W only: `{vx: 2.0, vy: 0.0, vz: 0.0}`
- W+A (forward-left): `{vx: 2.0, vy: -2.0, vz: 0.0}`
- E only (camera-up): `{vx: 0.0, vy: 0.0, vz: -2.0}` (negative = up in camera frame)
- No keys: `{vx: 0.0, vy: 0.0, vz: 0.0}`

---

### Helper Function: `_transform_to_ned(velocity_camera: Dict, attitude: Dict) -> Dict[str, float]`

**Purpose:** Rotate camera-relative 3D velocity to NED frame using full drone/camera attitude.

**Signature:**
```python
def _transform_to_ned(velocity_camera: Dict[str, float], attitude: Dict[str, float]) -> Dict[str, float]
```

**Inputs:**
- `velocity_camera`: `{'vx': float, 'vy': float, 'vz': float}` in camera frame (vx forward, vy right, vz down)
-- `attitude`: `{'roll': float, 'pitch': float, 'yaw': float}` in radians — this should be the *camera* attitude (body attitude + fixed camera mount offset + optional mouse deltas). Compute camera attitude from `armed_controller.data['attitude']` before calling this helper.

**Output:**
```python
{
    'vx': float,  # NED North velocity [m/s]
    'vy': float,  # NED East velocity [m/s]
    'vz': float   # NED Down velocity [m/s]
}
```

**Transformation:**
Use the full rotation matrix constructed from roll (φ), pitch (θ), yaw (ψ):

```
R = R_z(psi) * R_y(theta) * R_x(phi)
v_ned = R * v_cam
```

Where the elemental rotation matrices are defined as in the main transformation section above. This maps camera-local forward/right/down velocities into NED coordinates.

**Example (level camera, roll=0, pitch=0, yaw=0):**
- Input: `velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}`, attitude = {0,0,0}
- Result: `{'vx': 2.0, 'vy': 0.0, 'vz': 0.0}` ✓ moves North

**Example (pitched camera):**
- With non-zero pitch, an `E` (camera-up) input produces vertical plus horizontal components in NED according to the camera orientation.

---

## 4. DATA STRUCTURES & CONSTANTS

### Constants
```python
SPEED_LATERAL = 2.0      # m/s for WASD movement
SPEED_VERTICAL = 2.0     # m/s for E/C movement
CONTROL_LOOP_HZ = 250    # Hz (called every 4 ms)
CAMERA_PITCH_OFFSET = 20.0  # degrees; camera is mounted +20° pitch relative to body
# (Implementation should convert to radians when composing rotations)

# Key names (must match keyboard library)
KEY_FORWARD = 'w'
KEY_LEFT = 'a'
KEY_BACKWARD = 's'
KEY_RIGHT = 'd'
KEY_UP = 'e'
KEY_DOWN = 'c'
```

### Velocity Dict Structure
Passed to `controller.update_position_flight_control()`:
```python
{
    'vx': float,  # NED North [m/s]
    'vy': float,  # NED East [m/s]
    'vz': float   # NED Down [m/s]
}
```

### Attitude Dict Structure
Retrieved from `armed_controller.data['attitude']`:
```python
{
    'roll': float,    # Body roll in radians
    'pitch': float,   # Body pitch in radians
    'yaw': float,     # Body yaw (heading) in radians; 0=North
    # ... possibly more keys
}
```

Note: the values above represent the vehicle *body* attitude. Compute the *camera* attitude by applying the fixed camera mount offset (`CAMERA_PITCH_OFFSET`) and any optional mouse deltas before using the rotation in `_transform_to_ned()`.

---

## 5. ALGORITHM FLOWCHART

```
handle_user_input(armed_controller)
  ├─ keys = _get_pressed_keys()
  │   └─ Return dict: {'w': bool, 'a': bool, 's': bool, 'd': bool, 'e': bool, 'c': bool}
  │
  ├─ keys = _resolve_conflicts(keys)
  │   └─ If W and S both pressed, set both False
  │   └─ If A and D both pressed, set both False
  │   └─ If E and C both pressed, set both False
  │
  ├─ velocity_camera = _calculate_velocity_camera_frame(keys)
  │   └─ vx = 2*w - 2*s
  │   └─ vy = -2*a + 2*d
  │   └─ vz = 2*c - 2*e
  │
    ├─ attitude = armed_controller.data['attitude'] (handle missing keys; use roll,pitch,yaw)
  
    ├─ velocity_ned = _transform_to_ned(velocity_camera, attitude)
    │   └─ v_ned = R(roll,pitch,yaw) * v_cam  # full 3D rotation described above
  │
  └─ controller.update_position_flight_control(
        armed_controller.sim_conn,
        armed_controller.system_boot_ms,
        velocity_ned
     )
```

---

## 6. EDGE CASES & ERROR HANDLING

| Case | Scenario | Expected Behavior | Implementation |
|---|---|---|---|
| **No keys pressed** | All keys released | Velocity = (0, 0, 0); drone hovers | _get_pressed_keys() returns all False |
| **Conflicting forward/backward** | W and S both pressed | Both cancel; vx = 0 | _resolve_conflicts() sets both False |
| **Conflicting left/right** | A and D both pressed | Both cancel; vy = 0 | _resolve_conflicts() sets both False |
| **Conflicting altitude** | E and C both pressed | Both cancel; vz = 0 | _resolve_conflicts() sets both False |
| **Multiple conflicts** | W+S+A+D all pressed | All cancel; hover | All resolve to False in _resolve_conflicts() |
| **Rapid key toggle** | Key pressed/released faster than control cycle | Detected on next frame (within 4ms) | keyboard.is_pressed() polled every cycle |
| **Missing attitude (roll/pitch/yaw)** | `shared_data['attitude']` missing or incomplete | Assume roll=0,pitch=0,yaw=0; log warning | Try/except, default to neutral attitude with warning log |
| **Null MAVLink connection** | armed_controller.sim_conn is None | Silently return; no command sent | Guard check before calling controller API |
| **Missing system_boot_ms** | armed_controller.system_boot_ms is None | Raise ValueError; this is critical | Fail fast; log error and re-raise |
| **Keyboard library exception** | keyboard.is_pressed() raises exception | Log error; return all False; continue running | Try/except in _get_pressed_keys() |
| **First frame initialization** | First call to handle_user_input() | Velocity = (0, 0, 0) if no keys pressed | No state needed; stateless function |
| **Simultaneous W and conflicting D+A** | W pressed + D pressed + A pressed | W allowed; D+A conflict → both False; move forward-right | Resolved independently |

---

## 7. TESTING & VERIFICATION

### Unit Tests

**Test 1: Conflict Resolution**
```
Test: _resolve_conflicts()
Input: {'w': True, 's': True, 'a': False, 'd': False, 'e': False, 'c': False}
Expected: {'w': False, 's': False, 'a': False, 'd': False, 'e': False, 'c': False}
```

**Test 2: No Keys**
```
Test: _calculate_velocity_camera_frame()
Input: {'w': False, 'a': False, 's': False, 'd': False, 'e': False, 'c': False}
Expected: {'vx': 0.0, 'vy': 0.0, 'vz': 0.0}
```

**Test 3: Single Directions**
```
Test: _calculate_velocity_camera_frame()
- W only → {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}
- A only → {'vx': 0.0, 'vy': -2.0, 'vz': 0.0}
- E only → {'vx': 0.0, 'vy': 0.0, 'vz': -2.0}
```

**Test 4: Diagonal Movement**
```
Test: _calculate_velocity_camera_frame()
Input: {'w': True, 'a': True, 's': False, 'd': False, 'e': False, 'c': False}
Expected: {'vx': 2.0, 'vy': -2.0, 'vz': 0.0}
```

**Test 5: NED Transformation (Yaw = 0, pointing North)**
```
Test: _transform_to_ned()
Input: velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}, attitude = {'roll':0, 'pitch':0, 'yaw':0}
Expected: {'vx': 2.0, 'vy': 0.0, 'vz': 0.0} (moves North)
```

**Test 6: NED Transformation (Yaw = π/2, pointing East)**
```
Test: _transform_to_ned()
Input: velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}, attitude = {'roll':0, 'pitch':0, 'yaw':π/2}
Expected: {'vx': ≈0, 'vy': 2.0, 'vz': 0.0} (moves East)
```

**Test 7: NED Transformation (Yaw = π/4, 45° heading)**
```
Test: _transform_to_ned()
Input: velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}, attitude = {'roll':0, 'pitch':0, 'yaw':π/4}
Expected: {'vx': √2, 'vy': √2, 'vz': 0.0} (moves northeast at 45°)
```

**Test 8: Altitude with Level Camera (independent of yaw)**
```
Test: _transform_to_ned() with roll=0, pitch=0 and various yaw values
Input: E key (vz_cam = -2.0)
Expected: v_ned == {'vx': 0.0, 'vy': 0.0, 'vz': -2.0} for any yaw
```

**Test 9: Altitude Affected by Pitch/Roll**
```
Test: _transform_to_ned() with non-zero pitch (θ) or roll (φ)
Input: E key (vz_cam = -2.0), pitch = 30° (θ ≈ 0.524 rad)
Expected: v_ned has a negative vz component (up) plus non-zero vx/vy components corresponding to the camera orientation
```

### Integration Tests (with Simulator)

1. **Single Direction Movement**
   - Precondition: Drone armed and hovering
   - Action: Press W
   - Verify: Drone moves forward (in direction camera pointing) at ~2.0 m/s
   - Verify: vx velocity command sent to controller is forward in NED

2. **Diagonal Movement**
   - Action: Press W + A simultaneously
   - Verify: Drone moves forward-left at 45° angle
   - Verify: Magnitude of velocity vector ≈ √(2² + 2²) ≈ 2.83 m/s

3. **Conflict Cancellation**
   - Action: Press W + S simultaneously
   - Verify: Drone hovers (no forward velocity)
   - Verify: Drone hovers (no lateral velocity from A+D test)

4. **Heading-Relative Movement**
   - Precondition: Rotate drone 90° (yaw = π/2, pointing East)
   - Action: Press W
   - Verify: Drone moves East (not North)

5. **Altitude Movement (Camera-Relative)**
    - Action: Press E
    - Verify: Drone moves along camera-up direction at 2.0 m/s; with level camera this results in pure climb, with pitched camera this produces climb plus horizontal components
    - Action: Press C
    - Verify: Drone moves along camera-down direction at 2.0 m/s

6. **Rapid Key Changes**
   - Action: Rapidly toggle W on/off (press, release, press, release)
   - Verify: Drone responds smoothly without delays or jitter
   - Verify: Velocity updates every frame (250 Hz)

7. **Hover on Release**
   - Precondition: Drone moving forward (W pressed)
   - Action: Release W
   - Verify: Drone stops forward motion within ~4 ms (one frame)

### Manual Verification (Simulator/Flight)

1. **Responsiveness**: Feel of control (should be smooth and reactive at 250 Hz)
2. **FPS-Style Feel**: Movement matches player expectations from FPS games
3. **No Jitter**: Smooth acceleration/deceleration when changing inputs
4. **Intuitive Heading**: Camera rotation affects movement direction naturally
5. **Diagonal Smoothness**: 45° movements are smooth and predictable

---

## 8. SCOPE & CONSTRAINTS

### Included
- Keyboard input (WASD, E, C) with conflict resolution
- Camera-heading-relative horizontal movement (WASD)
- Camera-relative vertical movement (E, C)
- NED frame transformation
- 250 Hz control loop integration
- Instant velocity changes (no ramping)
- Continuous polling (handles key events at sub-frame granularity)

### Excluded (Future Enhancements)
- Mouse control for camera rotation — planned; specs reserve hooks for mouse-driven camera deltas applied to the camera attitude (roll/pitch/yaw).
- Autonomous movement integration (external/autonomy velocity commands) — planned; leave API hooks to merge or prioritize external velocity targets with live user input.

### Assumptions
- Drone is already armed (no arming logic in navigation.py)
- MAVLink attitude state is populated and available
- Flight controller can accept NED velocity commands
- No GPS/vision feedback required for basic movement
- Keyboard library (used by existing code) is available
