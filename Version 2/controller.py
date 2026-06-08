import time

from pymavlink import mavutil

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
PITCH_RATE = -0.3   # rad/s (negative = pitch forward)
ROLL_RATE  = 0.0
YAW_RATE   = 0.0
THRUST     = 0.6    # 0.0 - 1.0

RATES_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
)

def update_attitude_flight_control(mavlink_conn, system_boot_ms):
    now_ms = int(time.time() * 1000)

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
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],  # dummy quaternion (ignored)
        ROLL_RATE,
        PITCH_RATE,
        YAW_RATE,
        THRUST
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

def update_position_flight_control(mavlink_conn, system_boot_ms, vz=0.5, vx=0.0, vy=0.0):
    now_ms = int(time.time() * 1000)

    """
    Sets a desired vehicle position in a local north-east-down coordinate
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
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        VELOCITY_POSITION_MASK,
        0.0, 0, 0.0,    # ignored position NED
        vx, vy, vz,     # commanded velocity in NED frame
        0.0, 0, 0.0,    # ignored acceleration
        0,              # ignored yaw
        0.0             # ignored yaw rate
    )

# --------------------------------------------------------------------------------------
# Control Loop
# --------------------------------------------------------------------------------------

CONTROL_HZ = 250

# m/s commanded once the countdown reaches "GO!". Positive y in local NED moves
# the drone slowly to the right.
RIGHT_VELOCITY = 0.2

# If the simulator's boot clock jumps backwards by more than this, we assume the
# sim was reset/restarted and we should re-run the countdown.
RESET_DETECT_BACKWARD_MS = 2000

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        # y-axis velocity command used while flying after "GO!".
        self.right_velocity = 0.0
        # True during the countdown: command a flat zero velocity so the drone
        # stays parked on the start pad and the simulator doesn't flag a false
        # start. False once we're flying.
        self.holding = True
        # race_start_boot_time_ms of the race we already flew. Used to ignore the
        # previous run's race_status that lingers in shared data after a restart
        # so we don't count down / GO on a stale, already-finished race.
        self.last_handled_start_ms = None
        # last observed sim boot clock, shared across the countdown and flight
        # phases so a backwards jump (sim restart) is detected from anywhere.
        self.prev_sim_ms = None

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
            # flying: move right slowly, keep x and z stable.
            update_position_flight_control(self.sim_conn, self.system_boot_ms, 0.0, 0.0, self.right_velocity)

        time.sleep(1.0 / CONTROL_HZ)

    # -------------------------------
    # Hold still (don't move at all)
    #
    # Keeps the drone parked on the start pad during the countdown by commanding
    # zero velocity, so it can't trigger the simulator's false-start detection.
    # -------------------------------
    def hover(self):
        self.holding = True
        self.right_velocity = 0.0

    # -------------------------------
    # Fly right (move on y, keep x and z stable)
    # -------------------------------
    def fly_right(self):
        # release the hold so the drone starts moving right.
        self.holding = False
        self.right_velocity = RIGHT_VELOCITY

    # -------------------------------
    # Countdown: "3... 2... 1... GO!" synced to the simulator's race clock
    #
    # Reads the sim's race status (shared by MAVLinkRX) and counts down to the
    # server's scheduled race start time instead of a local timer. Keeps
    # streaming hover setpoints the whole time so the drone holds altitude,
    # then commands forward flight exactly when the race starts.
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
    # Fly forward until the simulator is reset / a new race is scheduled
    #
    # Keeps streaming the forward velocity setpoint while watching the sim's
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
                self.prev_sim_ms = now_ms
                return
            self.prev_sim_ms = now_ms

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
