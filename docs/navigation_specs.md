# COMPREHENSIVE NAVIGATION.PY SPECIFICATIONS

## Overview & Purpose

`navigation.py` implements FPS-style keyboard and mouse control for drone flight. The module translates user input (WASD for movement, E/C for altitude, mouse for camera rotation) into NED-frame velocity commands sent to the flight controller at 250 Hz. Movement is always relative to camera heading except for altitude (which is world-relative).

---

## 1. REQUIREMENTS

### Functional Requirements

| Requirement | Input | Output | Notes |
|---|---|---|---|
| **Forward/Backward** | W key (forward) / S key (backward) | Velocity in camera-forward direction at 2.0 m/s | Regardless of drone heading |
| **Left/Right Strafe** | A key (left) / D key (right) | Velocity perpendicular to camera-forward (±90°) at 2.0 m/s | Perpendicular to forward direction |
| **Altitude Control** | E key (up) / C key (down) | Vertical velocity at 2.0 m/s (or -2.0 for down) | World-relative, independent of heading |
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
| **Camera Heading Source** | `armed_controller.data['attitude']['yaw']` (radians) |
| **Coordinate Transform** | Camera-relative movement (WASD) → NED frame using yaw rotation |

---

## 2. COORDINATE FRAMES & MATHEMATICAL FOUNDATION

### NED Frame (Used by Controller API)
- **North (vx)**: Positive north, negative south
- **East (vy)**: Positive east, negative west  
- **Down (vz)**: Positive down, negative up (note: inverted from typical +Z up convention)
- **Yaw (ψ)**: Drone heading in radians; 0 = north, π/2 = east, π/-π = south, -π/2 = west

### Camera-Relative Frame (User Input WASD)
- **Forward (W)**: Direction camera is pointing
- **Left (A)**: 90° counterclockwise from forward
- **Backward (S)**: 180° from forward
- **Right (D)**: 90° clockwise from forward

### Transformation: Camera-Relative → NED
To convert camera-relative velocities to NED frame, apply 2D rotation by yaw angle:

```
vx_ned = vx_camera * cos(yaw) - vy_camera * sin(yaw)
vy_ned = vx_camera * sin(yaw) + vy_camera * cos(yaw)
```

**Example:** If drone is pointed East (yaw = π/2):
- User presses W (forward): camera_vel = (2, 0) → NED = (0, 2) ✓ moves East
- User presses D (right): camera_vel = (0, -2) → NED = (-2, 0) ✓ moves North (perpendicular)

### Altitude (vz)
- **E key**: vz = -2.0 (negative in NED = upward)
- **C key**: vz = +2.0 (positive in NED = downward)
- **No input**: vz = 0

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
    - `.data['attitude']`: Dict with keys `yaw` (float, radians), possibly `roll`, `pitch`
    - Other state keys (optional for this function)

**Outputs:**
- None (side effect: calls `controller.update_position_flight_control()`)

**Behavior:**
1. Poll all relevant keys (W, A, S, D, E, C)
2. Resolve conflicts (W+S → W cancels, A+D → D cancels)
3. Build velocity vector from remaining keys
4. Retrieve drone yaw from `armed_controller.data['attitude']['yaw']`
5. Transform camera-relative velocity to NED frame using yaw
6. Call `controller.update_position_flight_control(armed_controller.sim_conn, armed_controller.system_boot_ms, velocity_dict)`
7. Return (no output)

**Error Handling:**
- If `armed_controller.data['attitude']['yaw']` is missing: log warning, assume yaw=0
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
    'vz': float   # Altitude relative [m/s]; NED frame (positive=down)
}
```

**Calculation:**
- `vx = (2.0 if keys['w'] else 0.0) + (-2.0 if keys['s'] else 0.0)`
- `vy = (-2.0 if keys['a'] else 0.0) + (2.0 if keys['d'] else 0.0)`
- `vz = (2.0 if keys['c'] else 0.0) + (-2.0 if keys['e'] else 0.0)`

**Examples:**
- W only: `{vx: 2.0, vy: 0.0, vz: 0.0}`
- W+A (forward-left): `{vx: 2.0, vy: -2.0, vz: 0.0}`
- E only (ascending): `{vx: 0.0, vy: 0.0, vz: -2.0}` (negative = up in NED)
- No keys: `{vx: 0.0, vy: 0.0, vz: 0.0}`

---

### Helper Function: `_transform_to_ned(velocity_camera: Dict, yaw: float) -> Dict[str, float]`

**Purpose:** Rotate camera-relative velocity to NED frame using drone heading.

**Signature:**
```python
def _transform_to_ned(velocity_camera: Dict[str, float], yaw: float) -> Dict[str, float]
```

**Inputs:**
- `velocity_camera`: `{'vx': float, 'vy': float, 'vz': float}` in camera frame
- `yaw`: Drone heading in radians (from MAVLink attitude)

**Output:**
```python
{
    'vx': float,  # NED North velocity [m/s]
    'vy': float,  # NED East velocity [m/s]
    'vz': float   # NED Down velocity [m/s]
}
```

**Transformation:**
```python
vx_ned = velocity_camera['vx'] * cos(yaw) - velocity_camera['vy'] * sin(yaw)
vy_ned = velocity_camera['vx'] * sin(yaw) + velocity_camera['vy'] * cos(yaw)
vz_ned = velocity_camera['vz']  # Altitude is NOT rotated
```

**Example:** Drone yaw = 0 (pointing North), user presses W (forward):
- Input: `velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}`, yaw = 0
- cos(0) = 1, sin(0) = 0
- Result: `{'vx': 2.0, 'vy': 0.0, 'vz': 0.0}` ✓ moves North

**Example:** Drone yaw = π/2 (pointing East), user presses W (forward):
- Input: `velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}`, yaw = π/2
- cos(π/2) = 0, sin(π/2) = 1
- Result: `{'vx': 0.0, 'vy': 2.0, 'vz': 0.0}` ✓ moves East

---

## 4. DATA STRUCTURES & CONSTANTS

### Constants
```python
SPEED_LATERAL = 2.0      # m/s for WASD movement
SPEED_VERTICAL = 2.0     # m/s for E/C movement
CONTROL_LOOP_HZ = 250    # Hz (called every 4 ms)

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
    'yaw': float,     # Heading in radians; 0=North
    'roll': float,    # (optional for navigation.py)
    'pitch': float,   # (optional for navigation.py)
    # ... possibly more keys
}
```

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
  ├─ yaw = armed_controller.data['attitude']['yaw'] (handle missing key)
  │
  ├─ velocity_ned = _transform_to_ned(velocity_camera, yaw)
  │   └─ vx_ned = vx*cos(yaw) - vy*sin(yaw)
  │   └─ vy_ned = vx*sin(yaw) + vy*cos(yaw)
  │   └─ vz_ned = vz
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
| **Missing yaw in attitude** | shared_data['attitude']['yaw'] not present | Assume yaw = 0; log warning | Try/except, default to 0 with warning log |
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
Input: velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}, yaw = 0
Expected: {'vx': 2.0, 'vy': 0.0, 'vz': 0.0} (moves North)
```

**Test 6: NED Transformation (Yaw = π/2, pointing East)**
```
Test: _transform_to_ned()
Input: velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}, yaw = π/2
Expected: {'vx': ≈0, 'vy': 2.0, 'vz': 0.0} (moves East)
```

**Test 7: NED Transformation (Yaw = π/4, 45° heading)**
```
Test: _transform_to_ned()
Input: velocity_camera = {'vx': 2.0, 'vy': 0.0, 'vz': 0.0}, yaw = π/4
Expected: {'vx': √2, 'vy': √2, 'vz': 0.0} (moves northeast at 45°)
```

**Test 8: Altitude Always World-Relative**
```
Test: _transform_to_ned() with various yaws
E key pressed at yaw=0, π/2, π: should always be {'vx': 0, 'vy': 0, 'vz': -2.0}
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

5. **Altitude Movement**
   - Action: Press E
   - Verify: Drone climbs at 2.0 m/s (altitude independent of heading)
   - Action: Press C
   - Verify: Drone descends at 2.0 m/s

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
- World-relative vertical movement (E, C)
- NED frame transformation
- 250 Hz control loop integration
- Instant velocity changes (no ramping)
- Continuous polling (handles key events at sub-frame granularity)

### Excluded (Future Enhancements)
- Mouse control for camera rotation (separate feature)
- Acceleration/velocity ramping (smoother feel)
- Proportional control (analog stick support)
- Gimbal/camera stabilization
- Automatic return-to-home or waypoint navigation
- Collision avoidance
- Speed presets (slow/fast modes)

### Assumptions
- Drone is already armed (no arming logic in navigation.py)
- MAVLink attitude state is populated and available
- Flight controller can accept NED velocity commands
- No GPS/vision feedback required for basic movement
- Keyboard library (used by existing code) is available

---

## 9. POINTS ON SPEC-DRIVEN DEVELOPMENT (Educational)

When writing specs like this, include these elements:

1. **Clear Functional Requirements**: What inputs produce what outputs? Be specific (e.g., "2.0 m/s" not "fast").

2. **Non-Functional Requirements**: Frequency, latency, frame choice, speed ranges. Often more important than you think.

3. **Mathematical Foundation**: When transformations are involved (rotations, frame changes), spell out the math. Reduces ambiguity.

4. **Function Signatures**: Exact inputs, outputs, and data types. Implementer shouldn't have to guess.

5. **Edge Cases & Error Handling**: What breaks? How do you handle missing data, conflicts, or unusual states?

6. **Examples & Test Cases**: Walk through concrete scenarios. A good example is worth 1000 words of explanation.

7. **Scope Boundaries**: Explicitly say what's IN and what's OUT. Prevents scope creep.

8. **References to Existing Code**: Link to APIs, constants, and patterns already in the codebase.

9. **Verification/Testing Strategy**: How will you know it works? Specific test cases, not vague statements.
