"""Simulator race lifecycle with PnP/EKF/MPCC gate flight control."""

import math
import os
import time
from dataclasses import dataclass

from pymavlink import mavutil

from gate_ekf import GatePoseEKF
from mpcc import CasadiMPCC, MPCCConfig


MAVLINK_CMD_SIM_RESET = 31000
CONTROL_HZ = 100
RESET_DETECT_BACKWARD_MS = 2000
DETECTION_MAX_AGE_S = 0.75
# Single authoritative active-flight floor. Change this value in code when a
# different minimum collective is required. Pre-race ground hold is separate.
ACTIVE_FLIGHT_MIN_THRUST = 0.27


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def clamp_active_flight_thrust(thrust):
    return max(ACTIVE_FLIGHT_MIN_THRUST, float(thrust))


def euler_to_quaternion(roll_rad, pitch_rad, yaw_rad):
    cy, sy = math.cos(yaw_rad * 0.5), math.sin(yaw_rad * 0.5)
    cp, sp = math.cos(pitch_rad * 0.5), math.sin(pitch_rad * 0.5)
    cr, sr = math.cos(roll_rad * 0.5), math.sin(roll_rad * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


ANGLE_ATTITUDE_MASK = (
    mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
    | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
    | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
)
RATES_ATTITUDE_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE


def send_attitude(mavlink_conn, system_boot_ms, roll_deg, pitch_deg, yaw_deg, thrust):
    # Final send-point guard: no active-flight caller can bypass the minimum.
    thrust = clamp_active_flight_thrust(thrust)
    quaternion = euler_to_quaternion(
        math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    )
    mavlink_conn.mav.set_attitude_target_send(
        int(time.time() * 1000) - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        ANGLE_ATTITUDE_MASK,
        quaternion,
        0.0, 0.0, 0.0,
        thrust,
    )


def send_throttle_down(mavlink_conn, system_boot_ms):
    """Keep collective at zero so the simulator permits the countdown."""
    mavlink_conn.mav.set_attitude_target_send(
        int(time.time() * 1000) - system_boot_ms,
        mavlink_conn.target_system,
        mavlink_conn.target_component,
        RATES_ATTITUDE_MASK,
        [1, 0, 0, 0],
        0.0, 0.0, 0.0,
        0.0,
    )


@dataclass
class FlightCommand:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    thrust_correction: float = 0.0
    tracking: bool = False
    source: str = "rules"
    distance_m: float = None
    gate_vertical_bearing_deg: float = None
    gate_vertical_control_active: bool = False
    vertical_accel_down_mps2: float = None


class RuleBasedGateController:
    """PD-style visual rules adapted from the Version 10 controller."""

    CONFIDENCE_MIN = 0.35
    CRUISE_PITCH_DEG = -0.01
    SEARCH_PITCH_DEG = -0.01
    ROLL_GAIN_DEG = 4.0
    ROLL_RATE_GAIN_DEG = 0.4
    LATERAL_VELOCITY_GAIN_DEG = -1.0
    MAX_ROLL_DEG = 5.0
    YAW_ORIENTATION_GAIN = -0.03
    MAX_YAW_DEG = 2.0
    THRUST_GAIN = -0.22
    THRUST_RATE_GAIN = -0.025
    MAX_THRUST_CORRECTION = 0.18
    SMOOTH_ALPHA = 0.12

    def __init__(self):
        self.reset()

    def reset(self):
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.previous_frame_id = None
        self.previous_frame_time = None
        self.previous_x = None
        self.previous_y = None
        self.x_rate = 0.0
        self.y_rate = 0.0

    def _valid(self, detection):
        return bool(
            detection
            and detection.get("detected")
            and detection.get("confidence", 0.0) >= self.CONFIDENCE_MIN
            and time.time() - detection.get("time", 0.0) <= DETECTION_MAX_AGE_S
        )

    def update(self, detection, lateral_velocity=0.0):
        if not self._valid(detection):
            self.roll_deg *= 0.90
            self.yaw_deg *= 0.90
            self.pitch_deg += 0.05 * (self.SEARCH_PITCH_DEG - self.pitch_deg)
            return FlightCommand(
                self.roll_deg, self.pitch_deg, self.yaw_deg, 0.0, False
            )

        offset_x = float(detection.get("offset_x", 0.0))
        offset_y = float(detection.get("offset_y", 0.0))
        orientation = float(detection.get("orientation_deg", 0.0))
        frame_id = detection.get("frame_id")
        now = time.time()
        if frame_id != self.previous_frame_id:
            if self.previous_frame_time is not None:
                dt = max(0.01, now - self.previous_frame_time)
                raw_x_rate = (offset_x - self.previous_x) / dt
                raw_y_rate = (offset_y - self.previous_y) / dt
                self.x_rate += 0.35 * (raw_x_rate - self.x_rate)
                self.y_rate += 0.35 * (raw_y_rate - self.y_rate)
            self.previous_frame_id = frame_id
            self.previous_frame_time = now
            self.previous_x = offset_x
            self.previous_y = offset_y

        roll_target = (
            self.ROLL_GAIN_DEG * offset_x
            + self.ROLL_RATE_GAIN_DEG * self.x_rate
            + self.LATERAL_VELOCITY_GAIN_DEG * float(lateral_velocity)
        )
        size_ratio = float(detection.get("size_ratio", 0.0))
        roll_limit = 3.0 if size_ratio > 0.55 else self.MAX_ROLL_DEG
        roll_target = clamp(roll_target, -roll_limit, roll_limit)
        yaw_target = clamp(
            self.YAW_ORIENTATION_GAIN * orientation,
            -self.MAX_YAW_DEG,
            self.MAX_YAW_DEG,
        )
        thrust_correction = clamp(
            self.THRUST_GAIN * offset_y + self.THRUST_RATE_GAIN * self.y_rate,
            -self.MAX_THRUST_CORRECTION,
            self.MAX_THRUST_CORRECTION,
        )
        self.roll_deg += self.SMOOTH_ALPHA * (roll_target - self.roll_deg)
        self.yaw_deg += self.SMOOTH_ALPHA * (yaw_target - self.yaw_deg)
        self.pitch_deg += 0.06 * (self.CRUISE_PITCH_DEG - self.pitch_deg)
        return FlightCommand(
            self.roll_deg, self.pitch_deg, self.yaw_deg,
            thrust_correction, True,
        )


class MPCCGateController:
    """Convert the first CasADi MPCC acceleration into bounded attitude targets."""

    GRAVITY_MPS2 = 9.80665
    MAX_ROLL_DEG = 4.0
    CENTERED_ROLL_LIMIT_DEG = 0.5
    ROLL_CENTER_BEARING_DEG = 1.0
    ROLL_FULL_AUTHORITY_BEARING_DEG = 15.0
    ROLL_SMOOTH_ALPHA = 0.12
    MAX_PITCH_DEG = 0.01
    PITCH_SMOOTH_ALPHA = 0.10
    PITCH_BRAKE_ALPHA = 0.25
    MAX_YAW_DEG = 4.0
    GATE_VERTICAL_DEADBAND_DEG = 2.0
    MAX_VERTICAL_MEASUREMENT_AGE_S = 0.15
    YAW_BEARING_GAIN = 0.16
    SMOOTH_ALPHA = 0.22

    def __init__(self):
        debug_speed_kmh = float(os.environ.get("MPCC_DEBUG_SPEED_KMH", "5.0"))
        debug_speed_mps = debug_speed_kmh / 3.6
        debug_speed_lock = os.environ.get("MPCC_DEBUG_SPEED_LOCK", "1") != "0"
        target_speed = (
            debug_speed_mps
            if debug_speed_lock
            else float(os.environ.get("MPCC_TARGET_SPEED_MPS", "3.0"))
        )
        config = MPCCConfig(
            horizon=int(os.environ.get("MPCC_HORIZON", "8")),
            dt=float(os.environ.get("MPCC_DT", "0.08")),
            weight_contour=float(os.environ.get("MPCC_W_CONTOUR", "35.0")),
            weight_lag=float(os.environ.get("MPCC_W_LAG", "4.0")),
            weight_progress=float(os.environ.get("MPCC_W_PROGRESS", "1.0")),
            weight_speed=float(os.environ.get("MPCC_W_SPEED", "12.0")),
            max_lateral_accel=float(
                os.environ.get("MPCC_MAX_LATERAL_ACCEL_MPS2", "1.0")
            ),
            max_vertical_accel=float(
                os.environ.get("MPCC_MAX_VERTICAL_ACCEL_MPS2", "2.0")
            ),
            max_forward_accel=float(
                os.environ.get("MPCC_MAX_FORWARD_ACCEL_MPS2", "0.8")
            ),
            max_forward_decel=float(
                os.environ.get("MPCC_MAX_FORWARD_DECEL_MPS2", "1.0")
            ),
            max_lateral_speed=float(
                os.environ.get("MPCC_MAX_LATERAL_SPEED_MPS", str(debug_speed_mps))
            ),
            max_vertical_speed=float(
                os.environ.get("MPCC_MAX_VERTICAL_SPEED_MPS", str(debug_speed_mps))
            ),
            max_forward_speed=(
                debug_speed_mps
                if debug_speed_lock
                else float(os.environ.get("MPCC_MAX_FORWARD_SPEED_MPS", "5.0"))
            ),
            target_speed=target_speed,
            max_progress_rate=(
                debug_speed_mps
                if debug_speed_lock
                else float(os.environ.get("MPCC_MAX_PROGRESS_RATE_MPS", "5.0"))
            ),
            debug_speed_lock=debug_speed_lock,
        )
        self.optimizer = CasadiMPCC(config)
        self.reset()

    @property
    def available(self):
        return self.optimizer.available

    @property
    def error(self):
        return self.optimizer.error

    def reset(self):
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.thrust_correction = 0.0
        self.vertical_accel_down_mps2 = 0.0
        self.last_result = None

    @classmethod
    def vertical_measurement_is_fresh(cls, measurement_age_s):
        if measurement_age_s is None:
            return False
        age = float(measurement_age_s)
        return math.isfinite(age) and 0.0 <= age <= cls.MAX_VERTICAL_MEASUREMENT_AGE_S

    @classmethod
    def roll_limit_for_bearing(cls, horizontal_bearing_deg):
        bearing = abs(float(horizontal_bearing_deg))
        if not math.isfinite(bearing):
            return cls.CENTERED_ROLL_LIMIT_DEG
        if bearing <= cls.ROLL_CENTER_BEARING_DEG:
            return cls.CENTERED_ROLL_LIMIT_DEG
        span = (
            cls.ROLL_FULL_AUTHORITY_BEARING_DEG
            - cls.ROLL_CENTER_BEARING_DEG
        )
        blend = clamp(
            (bearing - cls.ROLL_CENTER_BEARING_DEG) / span,
            0.0,
            1.0,
        )
        return (
            cls.CENTERED_ROLL_LIMIT_DEG
            + blend * (cls.MAX_ROLL_DEG - cls.CENTERED_ROLL_LIMIT_DEG)
        )

    @classmethod
    def roll_target_from_acceleration(cls, lateral_accel_mps2, bearing_deg):
        # Body Y is positive right and MAVLink positive roll banks right, so
        # lateral acceleration and roll must have the same sign.
        target = math.degrees(
            math.atan2(float(lateral_accel_mps2), cls.GRAVITY_MPS2)
        )
        limit = cls.roll_limit_for_bearing(bearing_deg)
        return clamp(target, -limit, limit)

    @classmethod
    def pitch_target_from_acceleration(cls, forward_accel_mps2):
        # Positive forward acceleration requires a negative (nose-down) pitch.
        target = -math.degrees(
            math.atan2(float(forward_accel_mps2), cls.GRAVITY_MPS2)
        )
        return clamp(target, -cls.MAX_PITCH_DEG, cls.MAX_PITCH_DEG)

    def update(self, gate_track):
        if not gate_track or not gate_track.get("valid"):
            return None
        position = gate_track["body_position_m"]
        relative_velocity = gate_track["body_velocity_mps"]
        gate_vector = [position["x"], position["y"], position["z"]]
        # A stationary gate's relative velocity is the negative drone velocity.
        drone_velocity = [
            -relative_velocity["x"],
            -relative_velocity["y"],
            -relative_velocity["z"],
        ]
        result = self.optimizer.solve(gate_vector, drone_velocity)
        if result is None:
            self.last_result = {
                "mode": "solver_failure_safe_brake",
                "error": self.optimizer.error,
                "target_speed_mps": self.optimizer.config.target_speed,
            }
            return self._safe_brake_command(gate_track, drone_velocity[0])

        acceleration = result["acceleration_body_mps2"]
        pitch_target = self.pitch_target_from_acceleration(
            acceleration["x"]
        )
        bearing_data = gate_track.get("bearing_body_deg", {})
        bearing = bearing_data.get("horizontal", 0.0)
        vertical_bearing = float(bearing_data.get("vertical", 0.0))
        measurement_age = gate_track.get("measurement_age_s")
        vertical_control_active = self.vertical_measurement_is_fresh(measurement_age)
        roll_target = self.roll_target_from_acceleration(
            acceleration["y"], bearing
        )
        yaw_target = clamp(
            self.YAW_BEARING_GAIN * float(bearing),
            -self.MAX_YAW_DEG,
            self.MAX_YAW_DEG,
        )
        # Use MPCC's bounded vertical acceleration. It incorporates both gate
        # height and estimated vertical velocity, so it can brake a climb or a
        # descent instead of selecting thrust from height alone.
        vertical_accel_target = (
            float(acceleration["z"])
            if vertical_control_active
            else 0.0
        )

        alpha = 1.0 if result.get("mode") == "overspeed_brake" else self.SMOOTH_ALPHA
        roll_alpha = (
            1.0
            if result.get("mode") == "overspeed_brake"
            else self.ROLL_SMOOTH_ALPHA
        )
        self.roll_deg += roll_alpha * (roll_target - self.roll_deg)
        pitch_alpha = (
            self.PITCH_BRAKE_ALPHA
            if result.get("mode") == "overspeed_brake"
            else self.PITCH_SMOOTH_ALPHA
        )
        self.pitch_deg += pitch_alpha * (pitch_target - self.pitch_deg)
        self.yaw_deg += alpha * (yaw_target - self.yaw_deg)
        # Never continue a vertical command from an EKF-coasted gate after the
        # visual measurement disappears. Return vertical acceleration to zero.
        vertical_alpha = alpha if vertical_control_active else 1.0
        self.vertical_accel_down_mps2 += vertical_alpha * (
            vertical_accel_target - self.vertical_accel_down_mps2
        )
        self.thrust_correction = 0.0
        self.last_result = result
        return FlightCommand(
            roll_deg=self.roll_deg,
            pitch_deg=self.pitch_deg,
            yaw_deg=self.yaw_deg,
            thrust_correction=self.thrust_correction,
            tracking=True,
            source=f"{result.get('mode', 'mpcc')}/{gate_track.get('source', 'ekf')}",
            distance_m=gate_track.get("distance_m"),
            gate_vertical_bearing_deg=(
                vertical_bearing if vertical_control_active else None
            ),
            gate_vertical_control_active=vertical_control_active,
            vertical_accel_down_mps2=(
                self.vertical_accel_down_mps2 if vertical_control_active else None
            ),
        )

    def _safe_brake_command(self, gate_track, forward_speed):
        brake_pitch = self.MAX_PITCH_DEG if forward_speed > 0.25 else 0.0
        alpha = self.SMOOTH_ALPHA
        self.roll_deg *= 1.0 - alpha
        self.pitch_deg += alpha * (brake_pitch - self.pitch_deg)
        self.yaw_deg *= 1.0 - alpha
        self.thrust_correction = 0.0
        self.vertical_accel_down_mps2 = 0.0
        return FlightCommand(
            roll_deg=self.roll_deg,
            pitch_deg=self.pitch_deg,
            yaw_deg=self.yaw_deg,
            thrust_correction=self.thrust_correction,
            tracking=True,
            source="solver_failure_safe_brake",
            distance_m=gate_track.get("distance_m"),
        )


class Controller:
    # Initial gravity feed-forward used to translate acceleration into an
    # absolute MAVLink collective. It is not a thrust floor or gate target.
    INITIAL_HOVER_THRUST = 0.28
    MIN_THRUST = ACTIVE_FLIGHT_MIN_THRUST
    MAX_THRUST = 0.48
    MAX_COMMAND_PITCH_DEG = 0.01
    ALTITUDE_GAIN = 0.03

    def __init__(self, sim_conn, data, system_boot_ms):
        self.sim_conn = sim_conn
        self.data = data
        self.system_boot_ms = system_boot_ms
        self.holding = True
        self.hold_z = 0.0
        self.last_handled_start_ms = None
        self.previous_sim_ms = None
        # Retain only a broad actuator envelope; MPCC continuously selects the
        # value inside it. Zero pre-flight throttle uses a separate path.
        self.min_thrust = self.MIN_THRUST
        self.hover_thrust_estimate = float(
            os.environ.get(
                "INITIAL_HOVER_THRUST",
                os.environ.get("BASE_THRUST", str(self.INITIAL_HOVER_THRUST)),
            )
        )
        self.max_thrust = float(os.environ.get("MAX_THRUST", str(self.MAX_THRUST)))
        self.rules = RuleBasedGateController()
        self.gate_filter = GatePoseEKF(
            max_coast_s=float(os.environ.get("GATE_EKF_COAST_S", "1.25"))
        )
        self.imu_history = self.data.get("imu_buffer")
        self.mpcc = MPCCGateController()
        self.last_pose_frame_id = None
        self.last_status_print = 0.0
        self.status_interval_s = float(os.environ.get("TERMINAL_STATUS_INTERVAL_S", "0.75"))
        self.status_rows = 0
        self.data["mpcc_status"] = "ready" if self.mpcc.available else "unavailable"
        if not self.mpcc.available:
            self.data["mpcc_error"] = self.mpcc.error
            print(f"MPCC unavailable; using visual fallback: {self.mpcc.error}", flush=True)

    def _update_gate_track(self, detection):
        now = time.time()
        race = self.data.get("race_status") or {}
        gate_index = race.get("active_gate_index")
        self.gate_filter.set_gate_index(gate_index)

        frame_id = (detection or {}).get("frame_id")
        pose = (detection or {}).get("pose") or {}
        detection_time = float((detection or {}).get("time", 0.0))
        fresh = now - detection_time <= DETECTION_MAX_AGE_S
        if (
            frame_id is not None
            and frame_id != self.last_pose_frame_id
            and fresh
            and pose.get("valid")
        ):
            body_position = pose["body_position_m"]
            distance = float(pose.get("distance_m", 0.0))
            reprojection = float(pose.get("reprojection_error_px", 0.0))
            measurement_std = clamp(
                0.04 + 0.01 * distance + 0.015 * reprojection,
                0.05,
                0.75,
            )
            capture_time, alignment_source = self._frame_capture_time(
                detection, now
            )
            self.gate_filter.observe_delayed(
                [body_position["x"], body_position["y"], body_position["z"]],
                capture_timestamp=capture_time,
                present_timestamp=now,
                imu_history=self.imu_history,
                gate_index=gate_index,
                measurement_std=measurement_std,
                alignment_source=alignment_source,
            )
            self.last_pose_frame_id = frame_id

        track = self.gate_filter.estimate(now, self.imu_history)
        self.data["gate_track"] = track
        return track

    def _frame_capture_time(self, detection, now):
        raw_timestamp_ns = (detection or {}).get("timestamp_ns")
        candidates = []
        if raw_timestamp_ns is not None:
            raw_timestamp_ns = int(raw_timestamp_ns)
            if raw_timestamp_ns >= 1_000_000_000_000_000:
                offset_ns = (self.data.get("timesync") or {}).get(
                    "server_to_local_offset_ns"
                )
                if offset_ns is not None:
                    candidates.append((
                        (raw_timestamp_ns + int(offset_ns)) / 1e9,
                        "server_epoch+timesync",
                    ))
                candidates.append((raw_timestamp_ns / 1e9, "server_epoch"))
            else:
                boot_epoch_ns = self.data.get("sim_boot_epoch_ns")
                if boot_epoch_ns is not None:
                    candidates.append((
                        (int(boot_epoch_ns) + raw_timestamp_ns) / 1e9,
                        "sim_boot_epoch",
                    ))

        for candidate, source in candidates:
            age = now - candidate
            if -0.02 <= age <= 1.5:
                return min(candidate, now), source

        received_time_ns = (detection or {}).get("received_time_ns")
        if received_time_ns is not None:
            return min(float(received_time_ns) / 1e9, now), "frame_receive_fallback"
        inference_s = float((detection or {}).get("inference_ms", 0.0)) / 1000.0
        return min(float((detection or {}).get("time", now)) - inference_s, now), "inference_start_fallback"

    def update(self):
        cycle_started = time.perf_counter()
        if self.holding:
            send_throttle_down(self.sim_conn, self.system_boot_ms)
        else:
            velocity = self.data.get("local_velocity") or {}
            detection = self.data.get("gate_detection")
            gate_track = self._update_gate_track(detection)
            command = self.mpcc.update(gate_track) if self.mpcc.available else None
            self.data["mpcc_result"] = self.mpcc.last_result if command else None
            if command is None:
                command = self.rules.update(detection, velocity.get("y", 0.0))
            # Final actuator envelope applies to every controller path,
            # including visual fallback and emergency braking.
            command.pitch_deg = clamp(
                command.pitch_deg,
                -self.MAX_COMMAND_PITCH_DEG,
                self.MAX_COMMAND_PITCH_DEG,
            )
            local_position = self.data.get("local_position") or {}
            current_z = local_position.get("z", self.hold_z)
            altitude_correction = (current_z - self.hold_z) * self.ALTITUDE_GAIN
            # A fresh gate becomes the vertical reference. Do not let the old
            # takeoff-altitude hold fight a second gate at a different height.
            if command.gate_vertical_control_active:
                altitude_correction = 0.0
            elif command.thrust_correction < 0:
                altitude_correction = min(altitude_correction, 0.0)
            elif command.thrust_correction > 0:
                altitude_correction = max(altitude_correction, 0.0)
            thrust = self._calculate_collective(
                command,
                altitude_correction,
                altitude_observable="z" in local_position,
            )
            send_attitude(
                self.sim_conn, self.system_boot_ms,
                command.roll_deg, command.pitch_deg, command.yaw_deg, thrust,
            )
            self._print_status(command, thrust, detection)
        remaining = (1.0 / CONTROL_HZ) - (time.perf_counter() - cycle_started)
        if remaining > 0.0:
            time.sleep(remaining)

    @staticmethod
    def _tilt_compensated_hover(base_thrust, roll_deg, pitch_deg):
        vertical_fraction = (
            math.cos(math.radians(float(roll_deg)))
            * math.cos(math.radians(float(pitch_deg)))
        )
        return float(base_thrust) / max(vertical_fraction, 0.75)

    def _calculate_collective(
        self, command, altitude_correction=0.0, altitude_observable=False
    ):
        gravity_feedforward = self._tilt_compensated_hover(
            self.hover_thrust_estimate, command.roll_deg, command.pitch_deg
        )
        gate_below = (
            command.tracking
            and command.gate_vertical_control_active
            and command.gate_vertical_bearing_deg is not None
            and command.gate_vertical_bearing_deg
            > MPCCGateController.GATE_VERTICAL_DEADBAND_DEG
        )
        gate_above = (
            command.tracking
            and command.gate_vertical_control_active
            and command.gate_vertical_bearing_deg is not None
            and command.gate_vertical_bearing_deg
            < -MPCCGateController.GATE_VERTICAL_DEADBAND_DEG
        )
        vertical_accel = command.vertical_accel_down_mps2
        if command.gate_vertical_control_active and vertical_accel is not None:
            # Body Z and acceleration are positive down. For a linearized
            # multirotor, a_down = g * (1 - thrust/hover). Rearranging converts
            # MPCC's acceleration into the absolute normalized collective.
            acceleration_scale = (
                1.0
                - float(vertical_accel) / MPCCGateController.GRAVITY_MPS2
            )
            requested = (
                gravity_feedforward * acceleration_scale
                + float(altitude_correction)
            )
            vertical_source = "mpcc_acceleration"
        else:
            # With no fresh gate, retain only gravity feed-forward. No climb or
            # descent command is carried over from the previous gate.
            requested = (
                self.hover_thrust_estimate
                + float(altitude_correction)
                + command.thrust_correction
            )
            vertical_source = "gravity_feedforward"
        minimum_applied = requested < self.min_thrust
        thrust = clamp(requested, self.min_thrust, self.max_thrust)
        self.data["collective_status"] = {
            "hover_thrust_estimate": self.hover_thrust_estimate,
            "gravity_feedforward": gravity_feedforward,
            "requested": requested,
            "vertical_accel_down_mps2": vertical_accel,
            "vertical_source": vertical_source,
            "altitude_observable": bool(altitude_observable),
            "gate_below": gate_below,
            "gate_above": gate_above,
            "minimum_applied": minimum_applied,
            "thrust": thrust,
        }
        return thrust

    def _print_status(self, command, thrust, detection):
        if time.time() - self.last_status_print < self.status_interval_s:
            return
        self.last_status_print = time.time()
        confidence = (detection or {}).get("confidence", 0.0)
        state = command.source.upper() if command.tracking else "SEARCH"
        distance = "--" if command.distance_m is None else f"{command.distance_m:.1f}m"
        gate_track = self.data.get("gate_track") or {}
        ekf_speed = gate_track.get("ekf_closing_speed_mps")
        visual_speed = gate_track.get("raw_visual_closing_speed_mps")
        mpcc_result = self.data.get("mpcc_result") or {}
        target_speed = mpcc_result.get("target_speed_mps")
        measurement_rate = gate_track.get("raw_measurement_rate_hz")
        measurement_age = gate_track.get("measurement_age_s")
        vision_latency = gate_track.get("vision_latency_ms")
        replayed_samples = gate_track.get("replayed_imu_samples", 0)
        alignment = gate_track.get("timestamp_alignment", "--")
        imu_stats = (
            self.imu_history.stats(time.time()) if self.imu_history is not None else {}
        )
        detector = detection or {}
        collective = self.data.get("collective_status") or {}
        gate_height = self._format_gate_height(command.gate_vertical_bearing_deg)
        vertical_mode = (
            "MPCC-VERTICAL/MIN"
            if collective.get("minimum_applied")
            else "MPCC-VERTICAL"
            if command.gate_vertical_control_active
            else "NO-FRESH-GATE"
        )
        roi = detector.get("roi") or {}
        roi_label = "ROI" if roi.get("used") else "FULL"
        provider = str(detector.get("execution_provider", ""))
        runtime = (
            "CUDA"
            if provider == "CUDAExecutionProvider"
            else str(detector.get("runtime", "--")).upper()
        )
        status = (
            f"[FLIGHT] {state}\n"
            f"  Gate    confidence {confidence:>4.2f} | range {distance} | "
            f"height {gate_height}\n"
            f"  Speed   EKF {self._format_speed(ekf_speed):>10} | "
            f"visual {self._format_speed(visual_speed):>10} | "
            f"target {self._format_speed(target_speed):>10}\n"
            f"  Timing  vision {self._format_ms(detector.get('inference_ms')):>8} "
            f"({runtime}/{roi_label}) | capture lag {self._format_ms(vision_latency):>8} | "
            f"PnP {self._format_rate(measurement_rate):>7}, age {self._format_age(measurement_age):>6}\n"
            f"  Replay  {int(replayed_samples):>3} IMU samples | "
            f"IMU {self._format_rate(imu_stats.get('rate_hz')):>7} | clock {alignment}\n"
            f"  Command roll {command.roll_deg:+5.1f} | pitch {command.pitch_deg:+5.1f} | "
            f"yaw {command.yaw_deg:+5.1f} | thrust {thrust:.3f} | "
            f"az {self._format_accel(command.vertical_accel_down_mps2):>10} | "
            f"{vertical_mode}\n"
        )
        print(status, flush=True)
        pose = (detection or {}).get("pose") or {}
        if pose.get("valid") and os.environ.get("TERMINAL_VERBOSE_AXES", "0") == "1":
            camera = pose["camera_position_m"]
            body = pose["body_position_m"]
            opencv_camera = pose.get("opencv_camera_position_m") or {}
            print(
                "[PNP AXES] "
                f"CV(right,down,forward)=({opencv_camera.get('right', 0.0):+.2f},"
                f"{opencv_camera.get('down', 0.0):+.2f},"
                f"{opencv_camera.get('forward', 0.0):+.2f}) "
                f"C(forward,right,down)=({camera['x']:+.2f},"
                f"{camera['y']:+.2f},{camera['z']:+.2f}) "
                f"B(x-fwd,y-right,z-down)=({body['x']:+.2f},"
                f"{body['y']:+.2f},{body['z']:+.2f})",
                flush=True,
            )

    @staticmethod
    def _format_speed(value):
        return "--" if value is None else f"{float(value):+.2f}m/s"

    @staticmethod
    def _format_accel(value):
        return "--" if value is None else f"{float(value):+.2f}m/s2"

    @staticmethod
    def _format_gate_height(value):
        if value is None:
            return "--"
        value = float(value)
        if abs(value) <= MPCCGateController.GATE_VERTICAL_DEADBAND_DEG:
            return f"CENTER ({value:+.1f}deg)"
        location = "LOW" if value > 0.0 else "HIGH"
        return f"{location} ({value:+.1f}deg)"

    @staticmethod
    def _format_rate(value):
        return "--" if value is None else f"{float(value):.1f}Hz"

    @staticmethod
    def _format_age(value):
        return "--" if value is None else f"{float(value):.2f}s"

    @staticmethod
    def _format_ms(value):
        return "--" if value is None else f"{float(value):.1f}ms"

    def hover(self):
        self.holding = True
        if self.imu_history is not None:
            self.imu_history.start_calibration()

    def start_flying(self):
        self.hold_z = (self.data.get("local_position") or {}).get("z", 0.0)
        self.rules.reset()
        gate_index = (self.data.get("race_status") or {}).get("active_gate_index")
        self.gate_filter.reset(gate_index)
        self.mpcc.reset()
        self.last_pose_frame_id = None
        if self.imu_history is not None:
            self.imu_history.freeze_calibration()
        self.holding = False

    def arm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0,
        )

    def disarm(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def wait_until_armed(self, timeout_s=5.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.update()
            if (self.data.get("heartbeat") or {}).get("armed"):
                print("Drone armed.", flush=True)
                return True
        print("Arming confirmation timed out; waiting for race.", flush=True)
        return False

    def send_sim_reset_command(self):
        self.sim_conn.mav.command_long_send(
            self.sim_conn.target_system, self.sim_conn.target_component,
            MAVLINK_CMD_SIM_RESET,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def _clock_restarted(self, now_ms):
        restarted = (
            self.previous_sim_ms is not None
            and now_ms < self.previous_sim_ms - RESET_DETECT_BACKWARD_MS
        )
        self.previous_sim_ms = now_ms
        if restarted:
            self.last_handled_start_ms = None
        return restarted

    def run_countdown(self):
        """Wait for a fresh simulator race and print its final 3-2-1-GO."""
        self.hover()
        print("Waiting for race countdown from simulator...", flush=True)
        last_count = None
        seen_future_start = False
        while True:
            self.update()
            race = self.data.get("race_status")
            if not race:
                continue
            start_ms = race.get("race_start_boot_time_ms")
            now_ms = race.get("sim_boot_time_ms")
            if now_ms is None:
                continue
            if self._clock_restarted(now_ms):
                seen_future_start = False
                last_count = None
            if start_ms is None or start_ms < 0:
                self.last_handled_start_ms = None
                seen_future_start = False
                last_count = None
                continue
            if start_ms == self.last_handled_start_ms:
                continue
            remaining_ms = start_ms - now_ms
            if remaining_ms > 0:
                seen_future_start = True
            if not seen_future_start:
                continue
            if remaining_ms <= 0:
                self.last_handled_start_ms = start_ms
                break
            count = int(remaining_ms // 1000) + 1
            if 1 <= count <= 3 and count != last_count:
                print(f"{count}...", flush=True)
                last_count = count
        print("GO!", flush=True)
        self.start_flying()

    def fly_until_reset(self):
        """Fly until reset, restart, or a newly scheduled race is observed."""
        while True:
            self.update()
            race = self.data.get("race_status")
            if not race:
                continue
            start_ms = race.get("race_start_boot_time_ms")
            now_ms = race.get("sim_boot_time_ms")
            if now_ms is None:
                continue
            if self._clock_restarted(now_ms):
                self.hover()
                return
            if start_ms is None or start_ms < 0:
                self.last_handled_start_ms = None
                self.hover()
                return
            if start_ms != self.last_handled_start_ms and start_ms - now_ms > 0:
                self.hover()
                return
