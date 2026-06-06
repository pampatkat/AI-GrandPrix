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

# m/s commanded once the countdown reaches "GO!". Applied on the z axis only so
# the drone moves vertically while x and y stay stable (centered "in the
# middle"). NED z is positive-down, so NEGATIVE climbs UP and positive descends.
VERTICAL_VELOCITY = -0.2

# Keep holding the drone perfectly still for this many ms *after* "GO!" is
# announced before we let it start moving on z. The countdown reaches "GO!"
# exactly at the simulator's scheduled race start; this is the extra pause the
# drone waits (while still station-keeping) before climbing. Increase this to
# delay the start of movement further; decrease toward 0 to start sooner.
START_DELAY_MS = 5000

# If the simulator's boot clock jumps backwards by more than this, we assume the
# sim was reset/restarted and we should re-run the countdown.
RESET_DETECT_BACKWARD_MS = 2000

# Station-keeping gains used during the countdown to actively hold the drone on
# its locked spot. A static zero-velocity setpoint isn't enough - the drone
# drifts/sinks - so each tick we command a velocity that drives it back toward
# the locked position (a simple proportional controller on position error).
STATION_KEEP_KP = 1.5          # m/s of correction per metre of error
STATION_KEEP_MAX_SPEED = 1.0   # clamp the correction so it stays gentle

class Controller:
    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        # z-axis (vertical) velocity command used while flying after "GO!".
        self.vertical_velocity = 0.0
        # True during the countdown: actively hold position so the drone does
        # not move at all. False once we're flying.
        self.holding = True
        # NED position the drone is locked onto while holding (captured from the
        # first position fix). None until we get a fix.
        self.lock_position = None

    def update(self):
        # send automated targets to sim flight controller
        #update_attitude_flight_control(self.sim_conn, self.system_boot_ms)
        # alternatively one of
        #update_motor_control(self.sim_conn, self.system_boot_ms)

        if self.holding:
            # active station-keeping: drive the drone back toward its locked
            # spot so it stays put during the countdown (doesn't drift/sink).
            vx, vy, vz = self._station_keep_velocity()
            update_position_flight_control(self.sim_conn, self.system_boot_ms, vz, vx, vy)
        else:
            # flying: climb on z, keep x and y stable (vx=vy=0).
            update_position_flight_control(self.sim_conn, self.system_boot_ms, self.vertical_velocity)

        time.sleep(1.0 / CONTROL_HZ)

    # Compute the velocity that pulls the drone back to its locked position.
    # Falls back to a full stop (zero velocity) until we have a position fix.
    def _station_keep_velocity(self):
        pos = self.data.get('local_position')
        if pos is None:
            return 0.0, 0.0, 0.0
        if self.lock_position is None:
            self.lock_position = dict(pos)
        vx = self._clamp(STATION_KEEP_KP * (self.lock_position['x'] - pos['x']))
        vy = self._clamp(STATION_KEEP_KP * (self.lock_position['y'] - pos['y']))
        vz = self._clamp(STATION_KEEP_KP * (self.lock_position['z'] - pos['z']))
        return vx, vy, vz

    @staticmethod
    def _clamp(v):
        return max(-STATION_KEEP_MAX_SPEED, min(STATION_KEEP_MAX_SPEED, v))

    # -------------------------------
    # Hold still (don't move at all)
    #
    # Actively keeps the drone parked on its locked spot during the countdown.
    # -------------------------------
    def hover(self):
        self.holding = True
        self.vertical_velocity = 0.0

    # -------------------------------
    # Fly vertically (move on z, keep x and y stable)
    # -------------------------------
    def fly_forward(self):
        # release the hold and reset the lock so the next countdown re-latches.
        self.holding = False
        self.lock_position = None
        self.vertical_velocity = VERTICAL_VELOCITY

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

            # race not scheduled yet
            if start_ms is None or start_ms < 0:
                last_count = None
                seen_future_start = False
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
                break

            # only announce the final 3, 2, 1 seconds
            count = int(remaining_ms // 1000) + 1
            if 1 <= count <= 3 and count != last_count:
                print(f"{count}...", flush=True)
                last_count = count

        print("GO!", flush=True)

        # Stay perfectly still for START_DELAY_MS after "GO!" before moving:
        # keep station-keeping (holding) so the drone doesn't drift, then
        # release into vertical flight once the pause has elapsed.
        hold_until = time.time() + START_DELAY_MS / 1000.0
        while time.time() < hold_until:
            self.hover()
            self.update()

        self.fly_forward()

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
        prev_sim_ms = None
        while True:
            self.update()

            race = self.data.get('race_status')
            if race is None:
                continue

            start_ms = race['race_start_boot_time_ms']
            now_ms = race['sim_boot_time_ms']

            # sim clock went backwards -> the simulator was reset/restarted.
            if prev_sim_ms is not None and now_ms < prev_sim_ms - RESET_DETECT_BACKWARD_MS:
                return
            prev_sim_ms = now_ms

            # a new race got scheduled in the future while we were flying ->
            # the simulator started a new run, so go back to the countdown.
            if start_ms is not None and start_ms >= 0 and (start_ms - now_ms) > 0:
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
