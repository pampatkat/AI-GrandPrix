import math
import time

from pymavlink import mavutil

from gate_controller import GateController, GateControlConfig

# --------------------------------------------------------------------------------------
# RESET COMMAND
MAVLINK_CMD_SIM_RESET = 31000

# --------------------------------------------------------------------------------------
# MOTOR CONTROLS
# --------------------------------------------------------------------------------------

MOTOR_FRONT_LEFT = 0
MOTOR_FRONT_RIGHT = 0
MOTOR_BACK_LEFT = 0
MOTOR_BACK_RIGHT = 0

def update_motor_control(mavlink_conn, system_boot_ms):
    motor_rpms = [MOTOR_FRONT_LEFT, MOTOR_FRONT_RIGHT, MOTOR_BACK_LEFT, MOTOR_BACK_RIGHT, 0, 0, 0, 0]
    mavlink_conn.mav.set_actuator_control_target_send(
        int(time.time() * 1e6),
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        0,
        motor_rpms
    )

# --------------------------------------------------------------------------------------
# ATTITUDE CONTROLS
# --------------------------------------------------------------------------------------
# Forward speed is set by GateController.cruise_forward_pitch_deg (not here).
ROLL_DEG = -0.00
YAW_DEG = 0.0
BASE_THRUST = 0.30        # 0.0 - 1.0, lower cruise so the gate opening stays in view
ALTITUDE_THRUST_GAIN = 0.012
MIN_FLIGHT_THRUST = 0.15
MAX_FLIGHT_THRUST = 0.30
GATE_DETECTION_MAX_AGE_S = 0.35
# Lateral steering: gate left/right in the image -> roll to center on the opening.
GATE_ROLL_GAIN_DEG = -8.0
MAX_GATE_ROLL_DEG = 6
# Gate-frame tilt (orientation_deg): bank/yaw to align with the rotated gate.
# Raise *_GAIN for stronger turn; raise MAX_* to allow a larger correction cap.
GATE_ORIENTATION_ROLL_GAIN_DEG = 0.4
MAX_GATE_ORIENTATION_ROLL_DEG = 4.0
GATE_ORIENTATION_YAW_GAIN_DEG = 0.00
MAX_GATE_ORIENTATION_YAW_DEG = 0.5
GATE_VERTICAL_THRUST_GAIN = 0.20
# Level roll/yaw for this many seconds after the simulator reports a gate pass.
POST_GATE_STRAIGHTEN_S = 1.5
# Positive offset_y means the gate sits below image center (drone is too high).
GATE_LOW_IN_FRAME_THRESHOLD = 0.25
GATE_LOW_IN_FRAME_EXTRA_GAIN = 0.40

RATES_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
)

ANGLE_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE |
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
)

def euler_to_quaternion(roll_rad, pitch_rad, yaw_rad):
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)

    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]

def update_attitude_flight_control(
        mavlink_conn,
        system_boot_ms,
        thrust=BASE_THRUST,
        roll_deg=ROLL_DEG,
        pitch_deg=0.0,
        yaw_deg=YAW_DEG):
    now_ms = int(time.time() * 1000)
    attitude = euler_to_quaternion(
        math.radians(roll_deg),
        math.radians(pitch_deg),
        math.radians(yaw_deg)
    )

    """
    Sets a desired vehicle attitude. Used by an external controller to
    command the vehicle (manual controller or other system).
    
    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint8_t, values:ATTITUDE_TARGET_TYPEMASK)
    q                         : Attitude quaternion (w, x, y, z order, zero-rotation is 1, 0, 0, 0) (type:float)
    body_roll_rate            : Body roll rate [rad/s] (type:float)
    body_pitch_rate           : Body pitch rate [rad/s] (type:float)
    body_yaw_rate             : Body yaw rate [rad/s] (type:float)
    thrust                    : Collective thrust, normalized to 0 .. 1 (-1 .. 1 for vehicles capable of reverse trust) (type:float)
    """
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        ANGLE_ATTITUDE_MASK,
        attitude,
        0.0,     # roll rate ignored
        0.0,     # pitch rate ignored
        0.0,     # yaw rate ignored
        thrust
    )

def update_throttle_down(mavlink_conn, system_boot_ms):
    # Hold the drone on the start pad with the throttle fully down before "GO!".
    #
    # The simulator gates the race countdown behind a throttle-down check (it
    # refuses to start while the throttle is up). Streaming a zero-VELOCITY
    # position setpoint doesn't satisfy this: the autopilot still applies hover
    # throttle to hold altitude, so the sim sees the throttle up. Commanding a
    # zero-THRUST attitude target instead keeps the collective thrust at minimum
    # (throttle down) while still streaming setpoints so offboard control stays
    # ready for the moment we release into flight.
    now_ms = int(time.time() * 1000)
    mavlink_conn.mav.set_attitude_target_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],  # dummy quaternion (ignored)
        0.0,           # roll rate
        0.0,           # pitch rate
        0.0,           # yaw rate
        0.0            # thrust = throttle fully down
    )

# --------------------------------------------------------------------------------------
# POSITION CONTROLS
# --------------------------------------------------------------------------------------
VELOCITY_POSITION_MASK = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |

        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |

        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

def update_position_flight_control(mavlink_conn, system_boot_ms, vx=0.0, vy=0.0, vz=0.0):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle velocity in the drone body north-east-down coordinate
    frame. Used by an external controller to command the vehicle
    (manual controller or other system).

    time_boot_ms              : Timestamp (time since system boot). [ms] (type:uint32_t)
    target_system             : System ID (type:uint8_t)
    target_component          : Component ID (type:uint8_t)
    coordinate_frame          : Valid options are: MAV_FRAME_LOCAL_NED = 1, MAV_FRAME_LOCAL_OFFSET_NED = 7, MAV_FRAME_BODY_NED = 8, MAV_FRAME_BODY_OFFSET_NED = 9 (type:uint8_t, values:MAV_FRAME)
    type_mask                 : Bitmap to indicate which dimensions should be ignored by the vehicle. (type:uint16_t, values:POSITION_TARGET_TYPEMASK)
    x                         : X Position in NED frame [m] (type:float)
    y                         : Y Position in NED frame [m] (type:float)
    z                         : Z Position in NED frame (note, altitude is negative in NED) [m] (type:float)
    vx                        : X velocity in NED frame [m/s] (type:float)
    vy                        : Y velocity in NED frame [m/s] (type:float)
    vz                        : Z velocity in NED frame [m/s] (type:float)
    afx                       : X acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afy                       : Y acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    afz                       : Z acceleration or force (if bit 10 of type_mask is set) in NED frame in meter / s^2 or N [m/s/s] (type:float)
    yaw                       : yaw setpoint [rad] (type:float)
    yaw_rate                  : yaw rate setpoint [rad/s] (type:float)
    """
    mavlink_conn.mav.set_position_target_local_ned_send(
        now_ms - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        VELOCITY_POSITION_MASK,
        0.0, 0.0, 0.0,  # ignored position NED
        vx, vy, vz,     # commanded velocity: forward/right/down in body NED
        0.0, 0, 0.0,    # ignored acceleration
        0,              # ignored yaw
        0.0             # ignored yaw rate
    )

# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------

CONTROL_HZ = 250

# 15 km/h converted to m/s (unused in attitude+thrust mode; kept for reference).
FORWARD_VELOCITY = 15.0 / 3.6
VELOCITY_PRINT_INTERVAL_S = 0.5

# If the simulator's boot clock jumps backwards by more than this, we assume the
# sim was reset/restarted and we should re-run the countdown.
RESET_DETECT_BACKWARD_MS = 2000

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        # Thrust command used while flying after "GO!".
        self.flight_thrust = 0.0
        self.hold_z = 0.0
        # True during the countdown: command a flat zero velocity so the drone
        # stays parked on the start pad and the simulator doesn't flag a false
        # start. False once we're flying.
        self.holding = True
        # race_start_boot_time_ms of the race we already flew. Used to ignore the
        # current run's race_status after GO while still allowing a reset/restart
        # to reuse the same boot-relative start time.
        self.last_handled_start_ms = None
        # last observed sim boot clock, shared across the countdown and flight
        # phases so a backwards jump (sim restart) is detected from anywhere.
        self.prev_sim_ms = None
        self.roll_command_deg = ROLL_DEG
        self.pitch_command_deg = 0.0
        self.yaw_command_deg = YAW_DEG
        self.last_velocity_print_time = 0.0
        self.last_readiness_print_time = 0.0
        self.last_reported_active_gate_index = None
        self.last_reported_gate_pass_time = None
        self.flight_straightening = False
        # Robust gate follower
        self.gate_controller = GateController(GateControlConfig(
            roll_gain_deg=GATE_ROLL_GAIN_DEG,
            max_roll_near_deg=MAX_GATE_ROLL_DEG,
            orientation_roll_gain_deg=GATE_ORIENTATION_ROLL_GAIN_DEG,
            max_orientation_roll_deg=MAX_GATE_ORIENTATION_ROLL_DEG,
            orientation_yaw_gain_deg=GATE_ORIENTATION_YAW_GAIN_DEG,
            max_orientation_yaw_deg=MAX_GATE_ORIENTATION_YAW_DEG,
            thrust_gain=GATE_VERTICAL_THRUST_GAIN,
            low_in_frame_threshold=GATE_LOW_IN_FRAME_THRESHOLD,
            low_in_frame_extra_gain=GATE_LOW_IN_FRAME_EXTRA_GAIN,
            post_pass_straighten_s=POST_GATE_STRAIGHTEN_S,
            straighten_pitch_deg=0.0,
            confidence_min=0.45,
            detection_max_age_s=GATE_DETECTION_MAX_AGE_S,
        ))

    def update(self):
        # send automated targets to sim flight controller
        #update_attitude_flight_control(self.sim_conn, self.system_boot_ms)
        # alternatively one of
        #update_motor_control(self.sim_conn, self.system_boot_ms)

        if self.holding:
            # Keep the throttle fully down so the drone stays parked on the start
            # pad and the simulator's pre-race throttle-down check passes (a
            # zero-velocity hover setpoint doesn't - the autopilot still holds
            # altitude with hover throttle, which reads as "throttle up").
            update_throttle_down(self.sim_conn, self.system_boot_ms)
        else:
            current_z = (self.data.get('local_position') or {}).get('z', self.hold_z)
            # Feed the raw detection (the gate follower does its own confidence /
            # staleness filtering and gracefully coasts/decays through dropouts).
            detection = self.data.get('gate_detection')
            current_velocity_x = (self.data.get('local_velocity') or {}).get('x', 0.0)
            gate_command = self.gate_controller.update(
                detection,
                current_velocity_x=current_velocity_x,
            )
            self.roll_command_deg = gate_command.roll_deg
            self.pitch_command_deg = gate_command.pitch_deg
            self.yaw_command_deg = gate_command.yaw_deg
            self.flight_straightening = gate_command.straightening
            gate_thrust_correction = gate_command.thrust_correction

            # Keep exposing the (filtered) detection for the periodic printout.
            gate_detection = self.get_active_gate_detection()
            self.print_gate_pass_event()

            self.flight_thrust = self._clamp(
                BASE_THRUST
                + ((current_z - self.hold_z) * ALTITUDE_THRUST_GAIN)
                + gate_thrust_correction,
                MIN_FLIGHT_THRUST,
                MAX_FLIGHT_THRUST
            )

            # STABILIZE accepts attitude+thrust setpoints; velocity setpoints
            # are ignored (the drone falls). Forward speed comes from pitch tilt.
            update_attitude_flight_control(
                self.sim_conn,
                self.system_boot_ms,
                thrust=self.flight_thrust,
                roll_deg=self.roll_command_deg,
                pitch_deg=self.pitch_command_deg,
                yaw_deg=self.yaw_command_deg,
            )
            self.print_attitude_info(gate_detection)

        time.sleep(1.0 / CONTROL_HZ)

    # -------------------------------
    # Hold still (don't move at all)
    #
    # Keeps the drone parked on the start pad during the countdown by commanding
    # zero velocity, so it can't trigger the simulator's false-start detection.
    # -------------------------------
    def hover(self):
        self.holding = True
        self.flight_thrust = 0.0
        self.roll_command_deg = ROLL_DEG
        self.yaw_command_deg = YAW_DEG

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))

    def get_active_gate_detection(self):
        detection = self.data.get('gate_detection') or {}
        if not detection.get('detected'):
            return None
        if detection.get('confidence', 0.0) < 0.45:
            return None
        if time.time() - detection.get('time', 0.0) > GATE_DETECTION_MAX_AGE_S:
            return None
        return detection

    def wait_until_armed(self, timeout_s=3.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.update()
            heartbeat = self.data.get('heartbeat') or {}
            if heartbeat.get('armed'):
                self.print_readiness("Armed")
                return True

        self.print_readiness("Arming check timed out")
        return False

    def print_readiness(self, label):
        heartbeat = self.data.get('heartbeat') or {}
        ack = self.data.get('last_command_ack') or {}
        local_position = self.data.get('local_position') or {}
        local_velocity = self.data.get('local_velocity') or {}
        race = self.data.get('race_status') or {}
        ack_text = "none"
        if ack:
            ack_text = f"command={ack.get('command')} result={ack.get('result')}"
        print(
            f"\n[{label}]\n"
            f"  vehicle : armed={heartbeat.get('armed', 'unknown')}  "
            f"mode={heartbeat.get('mode', 'unknown')}  "
            f"status={heartbeat.get('system_status', 'unknown')}\n"
            f"  mavlink : base_mode={heartbeat.get('base_mode', 'unknown')}  "
            f"custom_mode={heartbeat.get('custom_mode', 'unknown')}  ack=({ack_text})\n"
            f"  position: x={local_position.get('x', 0.0): .2f}  "
            f"y={local_position.get('y', 0.0): .2f}  "
            f"z={local_position.get('z', 0.0): .2f}\n"
            f"  velocity: x={local_velocity.get('x', 0.0): .2f}  "
            f"y={local_velocity.get('y', 0.0): .2f}  "
            f"z={local_velocity.get('z', 0.0): .2f}\n"
            f"  race    : start_ms={race.get('race_start_boot_time_ms', 'unknown')}  "
            f"active_gate={race.get('active_gate_index', 'unknown')}  "
            f"last_gate_time={self.format_gate_race_time(race.get('last_gate_race_time'))}",
            flush=True
        )

    def print_attitude_info(self, gate_detection=None):
        now = time.time()
        if now - self.last_velocity_print_time < VELOCITY_PRINT_INTERVAL_S:
            return

        self.last_velocity_print_time = now
        local_velocity = self.data.get('local_velocity') or {}
        race = self.data.get('race_status') or {}
        gate_text = "none"
        if gate_detection:
            orient = gate_detection.get('orientation_deg')
            orient_text = f"  tilt={orient:+.1f}deg" if orient is not None else ""
            pnp_text = ""
            if gate_detection.get("pnp_valid"):
                pnp_text = (
                    f"  pnp_dist={gate_detection.get('pnp_distance_m', 0.0):.1f}m"
                )
            gate_text = (
                f"x={gate_detection.get('offset_x', 0.0):+.2f}  "
                f"y={gate_detection.get('offset_y', 0.0):+.2f}  "
                f"size={gate_detection.get('size_ratio', 0.0):.2f}  "
                f"conf={gate_detection.get('confidence', 0.0):.2f}"
                f"{orient_text}{pnp_text}"
            )
        pitch_mode = "straight" if self.flight_straightening else "cruise"
        print(
            "\n[Flight]\n"
            f"  command : pitch={self.pitch_command_deg: .1f} deg ({pitch_mode})  "
            f"roll={self.roll_command_deg: .1f} deg  "
            f"yaw={self.yaw_command_deg: .1f} deg  "
            f"thrust={self.flight_thrust:.2f}\n"
            f"  velocity: x={local_velocity.get('x', 0.0): .2f}  "
            f"y={local_velocity.get('y', 0.0): .2f}  "
            f"z={local_velocity.get('z', 0.0): .2f} m/s\n"
            f"  vision  : gate={gate_text}\n"
            f"  race    : active_gate={race.get('active_gate_index', 'unknown')}  "
            f"last_gate_time={self.format_gate_race_time(race.get('last_gate_race_time'))}",
            flush=True
        )

    def print_gate_pass_event(self):
        race = self.data.get('race_status') or {}
        active_gate_index = race.get('active_gate_index')
        last_gate_time = race.get('last_gate_race_time')
        if active_gate_index is None:
            return

        gate_time_changed = (
            active_gate_index > 0 and
            last_gate_time is not None
            and last_gate_time >= 0
            and last_gate_time != self.last_reported_gate_pass_time
        )
        gate_index_advanced = (
            self.last_reported_active_gate_index is not None
            and active_gate_index > self.last_reported_active_gate_index
        )
        if gate_time_changed or gate_index_advanced:
            passed_gate_index = max(0, active_gate_index - 1)
            self.gate_controller.notify_gate_passed()
            self.data["last_gate_pass_event"] = {
                "passed_gate_index": passed_gate_index,
                "next_gate_index": active_gate_index,
                "time": time.time(),
            }
            print(
                "\n[Gate Passed]\n"
                f"  simulator: passed_gate={passed_gate_index}  "
                f"next_gate={active_gate_index}  "
                f"race_time={self.format_gate_race_time(last_gate_time)}",
                flush=True
            )

        self.last_reported_active_gate_index = active_gate_index
        if last_gate_time is not None and last_gate_time >= 0:
            self.last_reported_gate_pass_time = last_gate_time

    @staticmethod
    def format_gate_race_time(raw_time):
        if raw_time is None or raw_time < 0:
            return "none"
        if raw_time >= 1_000_000_000:
            return f"{raw_time / 1_000_000_000.0:.3f}s"
        if raw_time >= 1_000:
            return f"{raw_time / 1_000.0:.3f}s"
        return f"{raw_time:.3f}s"

    # -------------------------------
    # Fly forward (move on x, keep y and z stable)
    # -------------------------------
    def fly_right(self):
        # Keep this method name so the existing GO path starts moving immediately.
        local_position = self.data.get('local_position') or {}
        self.hold_z = local_position.get('z', self.hold_z)
        self.holding = False
        self.flight_thrust = BASE_THRUST
        self.roll_command_deg = ROLL_DEG
        self.yaw_command_deg = YAW_DEG
        race = self.data.get('race_status') or {}
        self.last_reported_active_gate_index = race.get('active_gate_index')
        last_gate_time = race.get('last_gate_race_time')
        self.last_reported_gate_pass_time = (
            last_gate_time
            if last_gate_time is not None and last_gate_time >= 0
            else None
        )
        # Clear any tracking state carried over from a previous race so we start
        # the new run wings-level with no stale derivative/decay history.
        self.gate_controller.reset()
        self.print_readiness("GO readiness")

    # -------------------------------
    # Countdown: "3... 2... 1... GO!" synced to the simulator's race clock
    #
    # Reads the sim's race status (shared by MAVLinkRX) and counts down to the
    # server's scheduled race start time instead of a local timer. Keeps
    # streaming hover setpoints the whole time so the drone holds altitude,
        # then commands rightward flight exactly when the race starts.
    # -------------------------------
    def run_countdown(self):
        self.hover()
        print("Waiting for race start from simulator...", flush=True)

        last_count = None
        # Guards against an "imposter GO!": we only let the countdown finish
        # once we've actually seen the race scheduled in the FUTURE. Without
        # this, connecting mid-race (or a stale/already-elapsed start time
        # lingering from a previous run) would make remaining_ms <= 0 on the
        # very first loop and fire an instant GO with no 3... 2... 1...
        seen_future_start = False
        while True:
            # keep commanding a full stop (zero velocity on every axis) so the
            # drone doesn't move at all while the countdown is running.
            self.hover()
            self.update()

            race = self.data.get('race_status')
            if race is None:
                continue

            start_ms = race['race_start_boot_time_ms']
            now_ms = race['sim_boot_time_ms']

            # sim boot clock jumped backwards -> the simulator was restarted, so
            # whatever race we flew before is gone. Forget it (even a new race
            # that happens to reuse the same boot-relative start time is now
            # valid) and re-arm the countdown from scratch.
            if self.prev_sim_ms is not None and now_ms < self.prev_sim_ms - RESET_DETECT_BACKWARD_MS:
                self.last_handled_start_ms = None
                seen_future_start = False
                last_count = None
            self.prev_sim_ms = now_ms

            # race not scheduled yet
            if start_ms is None or start_ms < 0:
                # A reset can briefly clear the race schedule. Once that happens,
                # the next scheduled race should be allowed to GO even if the
                # simulator reuses the same boot-relative start time.
                self.last_handled_start_ms = None
                last_count = None
                seen_future_start = False
                continue

            # ignore the race we already flew: its race_status keeps arriving
            # (and lingers in shared data) after the flight, so without this the
            # countdown would instantly "GO" again on the stale, finished race.
            if start_ms == self.last_handled_start_ms:
                continue

            remaining_ms = start_ms - now_ms

            # latch once we've observed a genuine, not-yet-started race.
            if remaining_ms > 0:
                seen_future_start = True

            # ignore an already-elapsed start until we've seen a fresh race
            # scheduled in the future, so we don't fire an instant GO.
            if not seen_future_start:
                continue

            # the countdown ends exactly at the scheduled race start ("GO!").
            if remaining_ms <= 0:
                # remember this race so the flight phase (and the next countdown)
                # can tell it apart from a genuinely new one.
                self.last_handled_start_ms = start_ms
                break

            # only announce the final 3, 2, 1 seconds
            count = int(remaining_ms // 1000) + 1
            if 1 <= count <= 3 and count != last_count:
                print(f"{count}...", flush=True)
                last_count = count

        print("GO!", flush=True)

        self.fly_right()

    # -------------------------------
    # Fly right until the simulator is reset / a new race is scheduled
    #
    # Keeps streaming the rightward velocity setpoint while watching the sim's
    # race clock. Returns as soon as it detects the simulator was reset or
    # restarted so the caller can re-run the countdown. Detection covers:
    #   * the sim boot clock jumping backwards (sim rebooted), and
    #   * a fresh race being scheduled in the future while we're flying.
    # -------------------------------
    def fly_until_reset(self):
        while True:
            self.update()

            race = self.data.get('race_status')
            if race is None:
                continue

            start_ms = race['race_start_boot_time_ms']
            now_ms = race['sim_boot_time_ms']

            # sim clock went backwards -> the simulator was reset/restarted.
            if self.prev_sim_ms is not None and now_ms < self.prev_sim_ms - RESET_DETECT_BACKWARD_MS:
                self.last_handled_start_ms = None
                self.prev_sim_ms = now_ms
                return
            self.prev_sim_ms = now_ms

            # race schedule cleared -> simulator reset / waiting for a new run.
            if start_ms is None or start_ms < 0:
                self.last_handled_start_ms = None
                return

            # a genuinely new race (different from the one we just flew) got
            # scheduled in the future -> the simulator started a new run, so go
            # back to the countdown. Comparing against last_handled_start_ms
            # stops the race we're currently flying from being mistaken for a
            # new one.
            if (start_ms is not None and start_ms >= 0
                    and start_ms != self.last_handled_start_ms
                    and (start_ms - now_ms) > 0):
                return

    # -------------------------------
    # Arm the drone
    # -------------------------------
    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,  # arm
            0, 0, 0, 0, 0, 0
        )

    # -------------------------------
    # Disarm the drone
    # -------------------------------
    def disarm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,  # disarm
            0, 0, 0, 0, 0, 0
        )

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system,
            self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0,  # confirmation
            0, 0, 0, 0, 0, 0, 0
        )
