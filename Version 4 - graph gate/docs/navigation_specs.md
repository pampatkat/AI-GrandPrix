# COMPREHENSIVE NAVIGATION.PY SPECIFICATIONS

## Overview & Purpose

`navigation.py` implements FPS-style keyboard for drone flight. The module translates user input into attitude and thrust commands sent to the flight controller at 250 Hz. WASD modifies the vehicle attitude setpoint at a fixed rate, Q/E adjusts yaw heading, and V/C select fixed thrust setpoints relative to a hover baseline. ALL movement is still interpreted through the camera's orientation because the camera and vehicle body share the same origin; the camera is mounted tilted 20° upward relative to the body (camera pitch = body pitch + 20°). Autonomous attitude inputs are planned as future integrations and should be supported by the attitude composition described below.

---

## 1. REQUIREMENTS

### Functional Requirements

| Requirement | Input | Output | Notes |
|---|---|---|---|
| **Forward/Backward Attitude** | W key / S key | Pitch attitude command incremented/-decremented at a fixed rate | W/S tilt the vehicle nose down/up for forward/back motion |
| **Left/Right Attitude** | A key / D key | Roll attitude command incremented/-decremented at a fixed rate | A/D bank the vehicle left/right for lateral motion |
| **Yaw Heading** | Q key / E key | Yaw attitude command incremented/-decremented at a fixed step | Q/E rotate vehicle heading positively/negatively |
| **Altitude Thrust** | V key (up) / C key (down) | Fixed thrust setpoint above/below hover thrust | V/C select discrete climb/descend power levels |
| **Diagonal Input** | Any combination of WASD | Combined attitude targets across pitch and roll | Example: W+D produces pitch-down plus right-roll command |
| **Conflict Resolution** | Conflicting keys (W+S or A+D) | Conflicting attitude changes cancel to 0 on that axis | W+S → no pitch change; A+D → no roll change |
| **No Input** | No keys pressed | Hold hover thrust and level attitude | Vehicle should neither tilt nor change thrust from hover |
| **Continuous Polling** | Key events | State checked every control cycle (every 4ms) | Handles key press/release at sub-frame speed |
| **Camera-Aware Attitude** | Attitude inputs + camera mount offset | Body attitude commands are composed with camera pitch offset | Camera is tilted +20° relative to the body |

### Non-Functional Requirements

| Requirement | Specification |
|---|---|
| **Update Frequency** | Called every control cycle: 250 Hz (every 4 ms) |
| **Control Mode** | Attitude + thrust setpoints, not direct velocity commands |
| **Hover Thrust** | Base thrust is exactly the hover thrust required to balance gravity assuming starting attitude |
| **Thrust Values** | V/C select fixed thrust values relative to hover |
| **Attitude Rate** | WASD changes attitude at a fixed command rate or fixed incremental setpoint |
| **Response Latency** | <4ms (next control cycle at latest) |
| **Camera Attitude Source** | `armed_controller.data['attitude']['roll']`, `['pitch']`, `['yaw']` (radians) |
| **Reference Frame** | Attitude commands are expressed in the body/camera frame, with camera pitch offset applied |

---

## 2. ATTITUDE CONTROL FRAMEWORK

### Body/Camera Attitude
- The body and camera share the same origin.
- Camera attitude is derived from body attitude by applying a fixed camera mount offset:
    - `camera_roll = body_roll`
    - `camera_pitch = body_pitch + CAMERA_PITCH_OFFSET`
    - `camera_yaw = body_yaw`
- This means user-facing camera directions remain aligned with the camera pointing direction while the flight controller receives body attitude targets.

### WASD Attitude Mapping
- `W`/`S` adjust the pitch attitude target at a fixed command rate or incremental angle.
    - `W` tilts the nose down for forward movement.
    - `S` tilts the nose up for backward movement.
- `A`/`D` adjust the roll attitude target at a fixed command rate or incremental angle.
    - `A` rolls left for leftward motion.
    - `D` rolls right for rightward motion.
- Simultaneous WASD combinations produce combined pitch+roll attitude setpoints.
- Conflicting WASD keys on the same axis cancel each other.
- No WASD input means maintain the current level attitude target.

### V/C Thrust Mapping
- `V` selects a fixed thrust setpoint above hover thrust to climb.
- `C` selects a fixed thrust setpoint below hover thrust to descend.
- No V/C input holds hover thrust exactly equal to the thrust required to balance gravity.
- Thrust selection is discrete, not a direct vertical velocity command.

### Motion Semantics
- Forward/back motion is produced by changing pitch, not by commanding a forward velocity.
- Left/right motion is produced by changing roll, not by commanding a lateral velocity.
- Vertical motion is produced by thrust changes, not by commanding a vertical velocity.
- The controller receives attitude+thrust setpoints at 250 Hz.

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
- None (side effect: calls `controller.update_attitude_flight_control()`)

**Behavior:**
1. Poll all relevant keys (W, A, S, D, Q, E, V, C)
2. Resolve conflicts (W+S → both cancel, A+D → both cancel, Q+E → both cancel, V+C → both cancel)
3. Compute attitude targets from WASD input:
     - `W`/`S` adjust pitch at a fixed command rate or fixed incremental step
     - `A`/`D` adjust roll at a fixed command rate or fixed incremental step
     - `Q`/`E` adjust yaw heading at a fixed incremental step
4. Determine thrust setpoint from V/C:
     - no V/C → hover thrust exactly balancing gravity
     - V → fixed climb thrust above hover
     - C → fixed descend thrust below hover
5. Retrieve body attitude from `armed_controller.data['attitude']` (use roll, pitch, yaw)
     - Compute `camera_attitude` by applying the fixed camera mount offset (pitch +20°) and any optional mouse-driven deltas:
         `camera_attitude.roll = body_roll + mouse_roll_delta` (if provided)
         `camera_attitude.pitch = body_pitch + CAMERA_PITCH_OFFSET + mouse_pitch_delta`
         `camera_attitude.yaw = body_yaw + mouse_yaw_delta`
6. Call `controller.update_attitude_flight_control(armed_controller.sim_conn, armed_controller.system_boot_ms, thrust=thrust, roll_deg=roll_command, pitch_deg=pitch_command, yaw_deg=yaw_command)`
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
    'q': bool,
    'e': bool,
    'v': bool,
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
    'q': bool,
    'e': bool,
    'v': bool,
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
    'q': bool,  # False if both Q and E pressed
    'e': bool,  # False if both Q and E pressed
    'v': bool,  # False if both V and C pressed
    'c': bool   # False if both V and C pressed
}
```

**Conflict Resolution Rules:**
1. If `keys['w']` and `keys['s']` both True: set both to False
2. If `keys['a']` and `keys['d']` both True: set both to False
3. If `keys['q']` and `keys['e']` both True: set both to False (yaw conflict)
4. If `keys['v']` and `keys['c']` both True: set both to False (thrust conflict)
5. Return modified dict

**Example:**
- Input: `{'w': True, 's': True, 'a': False, 'd': True, 'q': False, 'e': False, 'v': False, 'c': False}`
- After resolution: `{'w': False, 's': False, 'a': False, 'd': True, 'q': False, 'e': False, 'v': False, 'c': False}`

---

### Helper Function: `_calculate_manual_attitude_command(keys: Dict[str, bool]) -> Tuple[float, float, float, float]`

**Purpose:** Convert resolved key presses into body attitude, yaw delta, and thrust targets.

**Signature:**
```python
def _calculate_manual_attitude_command(keys: Dict[str, bool]) -> Tuple[float, float, float, float]
```

**Input:** Resolved keys dict (from `_resolve_conflicts()`)

**Output:**
```python
{
    'roll_deg': float,      # commanded roll target in degrees
    'pitch_deg': float,     # commanded pitch target in degrees
    'yaw_delta_deg': float, # incremental yaw adjustment in degrees
    'thrust': float,        # normalized collective thrust [0.0 .. 1.0]
}
```

**Calculation:**
- `pitch_deg = forward_pitch_step` when `keys['w']`
- `pitch_deg = backward_pitch_step` when `keys['s']`
- `roll_deg = left_roll_step` when `keys['a']`
- `roll_deg = right_roll_step` when `keys['d']`
- `yaw_delta_deg = yaw_left_step` when `keys['q']`
- `yaw_delta_deg = yaw_right_step` when `keys['e']`
- `thrust = hover_thrust` when no `v`/`c` input
- `thrust = climb_thrust` when `keys['v']`
- `thrust = descend_thrust` when `keys['c']`
- No horizontal keys pressed leaves the corresponding attitude target at 0

**Examples:**
- W only: `pitch_deg = forward_pitch_step`, `roll_deg = 0`, `yaw_delta_deg = 0`
- A only: `pitch_deg = 0`, `roll_deg = left_roll_step`, `yaw_delta_deg = 0`
- Q only: `pitch_deg = 0`, `roll_deg = 0`, `yaw_delta_deg = yaw_left_step`
- V only: `thrust = climb_thrust`
- W+D: `pitch_deg = forward_pitch_step`, `roll_deg = right_roll_step`, `yaw_delta_deg = 0`
- No keys: all targets remain neutral and thrust = hover_thrust

---

### Thrust Computation

**Purpose:** Describe how V/C input is converted into the thrust component returned by `_calculate_manual_attitude_command()`.

**Behavior:**
- `thrust = hover_thrust` when no V/C keys are pressed
- `thrust = climb_thrust` when `keys['v']` is True
- `thrust = descend_thrust` when `keys['c']` is True
- V/C conflicts cancel and return `hover_thrust`

**Examples:**
- V only: climb thrust above hover
- C only: descend thrust below hover
- No keys: hover thrust
- V + C: hover thrust

---

## 4. DATA STRUCTURES & CONSTANTS

### Constants
```python
SPEED_LATERAL = 2.0      # nominal input scale for WASD attitude commands
SPEED_VERTICAL = 2.0     # nominal input scale for V/C thrust command behavior
CONTROL_LOOP_HZ = 250    # Hz (called every 4 ms)
CAMERA_PITCH_OFFSET = 20.0  # degrees; camera is mounted +20° pitch relative to body
# (Implementation should convert to radians when composing rotations)

# Key names (must match keyboard library)
KEY_FORWARD = 'w'
KEY_LEFT = 'a'
KEY_BACKWARD = 's'
KEY_RIGHT = 'd'
KEY_YAW_LEFT = 'q'
KEY_YAW_RIGHT = 'e'
KEY_UP = 'v'
KEY_DOWN = 'c'
```

### Control Setpoint Structure
Passed to `controller.update_attitude_flight_control()`:
```python
{
    'thrust': float,   # normalized collective thrust [0.0 .. 1.0]
    'roll_deg': float,  # body roll attitude target in degrees
    'pitch_deg': float, # body pitch attitude target in degrees
    'yaw_deg': float,   # body yaw attitude target in degrees
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

Note: the values above represent the vehicle *body* attitude. Compute the *camera* attitude by applying the fixed camera mount offset (`CAMERA_PITCH_OFFSET`) and any optional mouse deltas before composing attitude setpoints or feeding the controller.

---

## 5. ALGORITHM FLOWCHART

```
handle_user_input(armed_controller)
  ├─ keys = _get_pressed_keys()
  │   └─ Return dict: {'w': bool, 'a': bool, 's': bool, 'd': bool, 'q': bool, 'e': bool, 'v': bool, 'c': bool}
  │
  ├─ keys = _resolve_conflicts(keys)
  │   └─ Cancel W/S, A/D, Q/E, V/C conflicts on the same axis
  │
  ├─ attitude_command = _calculate_manual_attitude_command(keys)
  │   └─ pitch_command = fixed step from W/S
  │   └─ roll_command = fixed step from A/D
  │   └─ yaw_delta = fixed step from Q/E
  │   └─ thrust = hover/climb/descend from V/C
  │
  ├─ yaw_command = body_yaw + yaw_delta
  │
  └─ controller.update_attitude_flight_control(
        armed_controller.sim_conn,
        armed_controller.system_boot_ms,
        thrust=thrust_command,
        roll_deg=roll_command,
        pitch_deg=pitch_command,
        yaw_deg=yaw_command,
     )
```

---

## 6. EDGE CASES & ERROR HANDLING

| Case | Scenario | Expected Behavior | Implementation |
|---|---|---|---|
| **No keys pressed** | All keys released | Level attitude and hover thrust held; drone hovers | _get_pressed_keys() returns all False |
| **Conflicting forward/backward** | W and S both pressed | Both cancel; no pitch command change | _resolve_conflicts() sets both False |
| **Conflicting left/right** | A and D both pressed | Both cancel; no roll command change | _resolve_conflicts() sets both False |
| **Conflicting altitude** | V and C both pressed | Both cancel; hover thrust remains unchanged | _resolve_conflicts() sets both False |
| **Multiple conflicts** | W+S+A+D all pressed | All cancel; hover | All resolve to False in _resolve_conflicts() |
| **Rapid key toggle** | Key pressed/released faster than control cycle | Detected on next frame (within 4ms) | keyboard.is_pressed() polled every cycle |
| **Missing attitude (roll/pitch/yaw)** | `shared_data['attitude']` missing or incomplete | Assume roll=0,pitch=0,yaw=0; log warning | Try/except, default to neutral attitude with warning log |
| **Null MAVLink connection** | armed_controller.sim_conn is None | Silently return; no command sent | Guard check before calling controller API |
| **Missing system_boot_ms** | armed_controller.system_boot_ms is None | Raise ValueError; this is critical | Fail fast; log error and re-raise |
| **Keyboard library exception** | keyboard.is_pressed() raises exception | Log error; return all False; continue running | Try/except in _get_pressed_keys() |
| **First frame initialization** | First call to handle_user_input() | Level attitude and hover thrust held if no keys pressed | No state needed; stateless function |
| **Simultaneous W and conflicting D+A** | W pressed + D pressed + A pressed | W allowed; D+A conflict → both False; move forward-right | Resolved independently |

---

## 7. TESTING & VERIFICATION

### Unit Tests

**Test 1: Conflict Resolution**
```
Test: _resolve_conflicts()
Input: {'w': True, 's': True, 'a': False, 'd': False, 'q': False, 'e': False, 'v': False, 'c': False}
Expected: {'w': False, 's': False, 'a': False, 'd': False, 'q': False, 'e': False, 'v': False, 'c': False}
```

**Test 2: Attitude Command Generation**
```
Test: _calculate_manual_attitude_command()
Input: {'w': True, 'a': False, 's': False, 'd': False, 'q': False, 'e': False, 'v': False, 'c': False}
Expected: pitch_command = forward pitch step, roll_command = 0, yaw_delta = 0
```

**Test 3: Roll Command Generation**
```
Test: _calculate_manual_attitude_command()
Input: {'w': False, 'a': True, 's': False, 'd': False, 'q': False, 'e': False, 'v': False, 'c': False}
Expected: roll_command = left roll step, pitch_command = 0, yaw_delta = 0
```

**Test 4: Thrust Mode Selection**
```
Test: _calculate_manual_attitude_command()
Input: {'w': False, 'a': False, 's': False, 'd': False, 'q': False, 'e': False, 'v': True, 'c': False}
Expected: thrust = fixed climb thrust above hover
```

**Test 5: Hover Thrust**
```
Test: _calculate_manual_attitude_command()
Input: {'w': False, 'a': False, 's': False, 'd': False, 'q': False, 'e': False, 'v': False, 'c': False}
Expected: thrust = exact hover thrust
```

**Test 6: Combined Attitude**
```
Test: _calculate_manual_attitude_command()
Input: {'w': True, 'd': True, 's': False, 'a': False, 'q': False, 'e': False, 'v': False, 'c': False}
Expected: pitch_command = forward pitch step, roll_command = right roll step, yaw_delta = 0
```

**Test 7: No Input**
```
Test: handle_user_input()
Input: no keys pressed
Expected: controller.update_attitude_flight_control() called with hover thrust and roll/pitch targets of 0
```

**Test 8: Conflict Cancellation on Attitude**
```
Test: _resolve_conflicts()
Input: {'w': True, 's': True, 'a': True, 'd': True, 'q': True, 'e': True, 'v': True, 'c': True}
Expected: all keys cancel, no attitude or thrust change
```

**Test 9: Camera Offset Preservation**
```
Test: body attitude composition
Input: body_pitch = 0.0, CAMERA_PITCH_OFFSET = 20.0
Expected: camera_pitch = 20.0 degrees above body pitch
```

### Integration Tests (with Simulator)

1. **Single Direction Movement**
   - Precondition: Drone armed and hovering
   - Action: Press W
   - Verify: Drone moves forward in the camera-facing direction
   - Verify: controller.update_attitude_flight_control() is called with a forward pitch target and current yaw

2. **Diagonal Movement**
   - Action: Press W + A simultaneously
   - Verify: controller.update_attitude_flight_control() is called with both forward pitch and left roll targets
   - Verify: The pilot input results in a combined attitude command that produces forward-left motion

3. **Conflict Cancellation**
   - Action: Press W + S simultaneously
   - Verify: Drone holds hover thrust and maintains level attitude
   - Verify: No pitch or roll attitude command is generated from W+S conflict

4. **Heading-Relative Movement**
   - Precondition: Rotate drone 90° (yaw = π/2, pointing East)
   - Action: Press W
   - Verify: Drone moves East (not North)

5. **Altitude Movement (Camera-Relative)**
    - Action: Press V
    - Verify: Drone climbs with the camera pointing direction factored into ascent behavior; with level camera this results in a pure climb, with pitched camera this produces climb plus horizontal components
    - Action: Press C
    - Verify: Drone descends while respecting the camera orientation and thrust setpoint

6. **Rapid Key Changes**
   - Action: Rapidly toggle W on/off (press, release, press, release)
   - Verify: Drone responds smoothly without delays or jitter
   - Verify: attitude/thrust setpoints update every frame (250 Hz)

7. **Hover on Release**
   - Precondition: Drone moving forward (W pressed)
   - Action: Release W
   - Verify: Drone stops forward motion within ~4 ms (one frame)

### Manual Verification (Simulator/Flight)

1. **Responsiveness**: Feel of control (should be smooth and reactive at 250 Hz)
2. **Intuitive Heading**: Camera rotation affects movement direction naturally
3. **Diagonal Smoothness**: 45° movements are smooth and predictable

---

## 8. SCOPE & CONSTRAINTS

### Included
- Keyboard input (WASD, Q/E, V/C) with conflict resolution
- Camera-heading-relative horizontal movement (WASD)
- Camera-relative vertical movement (V, C)
- Attitude setpoint composition with camera pitch offset
- 250 Hz control loop integration
- Continuous polling (handles key events at sub-frame granularity)

### Excluded (Future Enhancements)
- Autonomous movement integration (external/autonomy attitude/thrust commands) — planned; leave API hooks to merge or prioritize external attitude targets with live user input.

### Assumptions
- Drone is already armed (no arming logic in navigation.py)
- MAVLink attitude state is populated and available
- No GPS/vision feedback required for basic movement
- Keyboard library (used by existing code) is available
