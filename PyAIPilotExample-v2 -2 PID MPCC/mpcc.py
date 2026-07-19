"""CasADi model-predictive contouring control objective for gate flight."""

import math
from dataclasses import dataclass

import numpy as np

try:
    import casadi as ca
except ImportError:  # setup can still explain how to install the optional runtime
    ca = None


@dataclass(frozen=True)
class MPCCConfig:
    horizon: int = 8
    dt: float = 0.08
    weight_contour: float = 35.0
    weight_lag: float = 4.0
    # Kept deliberately modest while the 5 km/h diagnostic lock is active.
    weight_progress: float = 1.0
    weight_speed: float = 12.0
    weight_acceleration: float = 0.08
    weight_acceleration_rate: float = 0.15
    max_lateral_accel: float = 1.0
    max_vertical_accel: float = 2.0
    max_forward_accel: float = 0.8
    max_forward_decel: float = 1.0
    max_lateral_speed: float = 1.3888888889
    max_vertical_speed: float = 1.3888888889
    max_forward_speed: float = 1.3888888889
    target_speed: float = 1.3888888889
    max_progress_rate: float = 1.3888888889
    debug_speed_lock: bool = True


class CasadiMPCC:
    """Optimize acceleration and virtual progress along a gate-center path."""

    def __init__(self, config=None):
        self.config = config or MPCCConfig()
        self.available = ca is not None
        self.error = None if self.available else "CasADi is not installed"
        self._solver = None
        self._guess = None
        if self.available:
            try:
                self._build_solver()
            except Exception as exc:
                self.available = False
                self.error = str(exc)

    def _build_solver(self):
        cfg = self.config
        horizon = int(cfg.horizon)
        if horizon < 2 or cfg.dt <= 0.0:
            raise ValueError("MPCC horizon and timestep must be positive")

        # Decision at each stage: body acceleration [ax, ay, az] and ds/dt.
        controls = ca.MX.sym("controls", 4, horizon)
        # Parameters: initial drone position/velocity, unit path tangent, s0.
        parameters = ca.MX.sym("parameters", 10)
        position = parameters[0:3]
        velocity = parameters[3:6]
        tangent = parameters[6:9]
        progress = parameters[9]
        objective = 0.0
        previous_acceleration = ca.MX.zeros(3, 1)
        velocity_constraints = []

        for stage in range(horizon):
            acceleration = controls[0:3, stage]
            progress_rate = controls[3, stage]
            reference = tangent * progress
            path_error = position - reference
            lag_error = ca.dot(tangent, path_error)
            contour_error = path_error - tangent * lag_error
            delta_progress = progress_rate * cfg.dt
            path_speed = ca.dot(tangent, velocity)

            # Requested MPCC objective:
            # Wc*e_c^2 + Wl*e_l^2 - Wp*Delta_s.
            objective += (
                cfg.weight_contour * ca.dot(contour_error, contour_error)
                + cfg.weight_lag * lag_error * lag_error
                - cfg.weight_progress * delta_progress
                + cfg.weight_speed * (path_speed - cfg.target_speed) ** 2
                + cfg.weight_acceleration * ca.dot(acceleration, acceleration)
                + cfg.weight_acceleration_rate
                * ca.dot(acceleration - previous_acceleration,
                         acceleration - previous_acceleration)
            )

            position = (
                position + velocity * cfg.dt
                + 0.5 * acceleration * (cfg.dt ** 2)
            )
            velocity = velocity + acceleration * cfg.dt
            # A hard physical state boundary at every prediction stage. These
            # constraints, unlike cost weights, cannot be traded for progress.
            velocity_constraints.append(velocity)
            progress = progress + delta_progress
            previous_acceleration = acceleration

        decision_vector = ca.reshape(controls, -1, 1)
        problem = {
            "x": decision_vector,
            "p": parameters,
            "f": objective,
            "g": ca.vertcat(*velocity_constraints),
        }
        options = {
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 35,
            "ipopt.tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
        }
        self._solver = ca.nlpsol("gate_mpcc", "ipopt", problem, options)

        progress_lower = cfg.target_speed if cfg.debug_speed_lock else 0.0
        progress_upper = (
            cfg.target_speed if cfg.debug_speed_lock else cfg.max_progress_rate
        )
        lower_stage = np.array(
            [
                -cfg.max_forward_decel,
                -cfg.max_lateral_accel,
                -cfg.max_vertical_accel,
                progress_lower,
            ],
            dtype=np.float64,
        )
        upper_stage = np.array(
            [
                cfg.max_forward_accel,
                cfg.max_lateral_accel,
                cfg.max_vertical_accel,
                progress_upper,
            ],
            dtype=np.float64,
        )
        self._lower_bounds = np.tile(lower_stage, horizon)
        self._upper_bounds = np.tile(upper_stage, horizon)
        velocity_lower = np.array(
            [-cfg.max_forward_speed, -cfg.max_lateral_speed, -cfg.max_vertical_speed],
            dtype=np.float64,
        )
        velocity_upper = -velocity_lower
        self._constraint_lower_bounds = np.tile(velocity_lower, horizon)
        self._constraint_upper_bounds = np.tile(velocity_upper, horizon)
        self._guess = np.zeros(4 * horizon, dtype=np.float64)
        self._guess[3::4] = cfg.target_speed if cfg.debug_speed_lock else 0.0

    @staticmethod
    def contour_and_lag_error(position, tangent, progress):
        position = np.asarray(position, dtype=np.float64).reshape(3)
        tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-9:
            raise ValueError("Path tangent cannot be zero")
        tangent = tangent / norm
        path_error = position - tangent * float(progress)
        lag_error = float(np.dot(tangent, path_error))
        contour_error = path_error - tangent * lag_error
        return contour_error, lag_error

    def solve(self, gate_position, drone_velocity=None):
        if not self.available:
            return None
        gate_position = np.asarray(gate_position, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(gate_position))
        if not np.all(np.isfinite(gate_position)) or distance < 1e-3:
            return None
        tangent = gate_position / distance
        velocity = np.zeros(3, dtype=np.float64) if drone_velocity is None else np.asarray(
            drone_velocity, dtype=np.float64
        ).reshape(3)
        if not np.all(np.isfinite(velocity)):
            return None

        speed_limits = np.array(
            [
                self.config.max_forward_speed,
                self.config.max_lateral_speed,
              self.config.max_vertical_speed,
            ],
            dtype=np.float64,
        )
        if np.any(np.abs(velocity) > speed_limits + 1e-6):
            return self._speed_guard_result(velocity, tangent, speed_limits)
        parameters = np.concatenate((np.zeros(3), velocity, tangent, [0.0]))

        try:
            solution = self._solver(
                x0=self._guess,
                p=parameters,
                lbx=self._lower_bounds,
                ubx=self._upper_bounds,
                lbg=self._constraint_lower_bounds,
                ubg=self._constraint_upper_bounds,
            )
            decision = np.asarray(solution["x"], dtype=np.float64).reshape(-1)
            self._guess = np.roll(decision, -4)
            self._guess[-4:] = decision[-4:]
            acceleration = decision[0:3]
            progress_rate = float(decision[3])
            control_stages = decision.reshape(self.config.horizon, 4)
            rollout_velocity = velocity.copy()
            peak_velocity = np.abs(rollout_velocity)
            for stage_controls in control_stages:
                rollout_velocity = (
                    rollout_velocity + stage_controls[0:3] * self.config.dt
                )
                peak_velocity = np.maximum(peak_velocity, np.abs(rollout_velocity))
            next_position = (
                velocity * self.config.dt
                + 0.5 * acceleration * self.config.dt ** 2
            )
            next_progress = progress_rate * self.config.dt
            next_velocity = velocity + acceleration * self.config.dt
            contour, lag = self.contour_and_lag_error(
                next_position, tangent, next_progress
            )
            return {
                "acceleration_body_mps2": {
                    "x": float(acceleration[0]),
                    "y": float(acceleration[1]),
                    "z": float(acceleration[2]),
                },
                "predicted_velocity_body_mps": {
                    "x": float(next_velocity[0]),
                    "y": float(next_velocity[1]),
                    "z": float(next_velocity[2]),
                },
                "predicted_peak_abs_velocity_mps": {
                    "x": float(peak_velocity[0]),
                    "y": float(peak_velocity[1]),
                    "z": float(peak_velocity[2]),
                },
                "mode": "optimized_speed_lock" if self.config.debug_speed_lock else "optimized",
                "speed_limit_mps": {
                    "x": self.config.max_forward_speed,
                    "y": self.config.max_lateral_speed,
                    "z": self.config.max_vertical_speed,
                },
                "target_speed_mps": self.config.target_speed,
                "progress_rate_mps": progress_rate,
                "delta_s_m": next_progress,
                "contour_error_m": float(np.linalg.norm(contour)),
                "lag_error_m": lag,
                "objective": float(solution["f"]),
                "path_tangent_body": {
                    "x": float(tangent[0]),
                    "y": float(tangent[1]),
                    "z": float(tangent[2]),
                },
            }
        except Exception as exc:
            self.error = str(exc)
            return None

    def _speed_guard_result(self, velocity, tangent, speed_limits):
        """Apply bounded deceleration if an input estimate is already overspeed."""
        cfg = self.config
        desired_acceleration = -velocity / max(cfg.dt, 1e-3)
        lower_acceleration = np.array(
            [-cfg.max_forward_decel, -cfg.max_lateral_accel, -cfg.max_vertical_accel],
            dtype=np.float64,
        )
        upper_acceleration = np.array(
            [cfg.max_forward_accel, cfg.max_lateral_accel, cfg.max_vertical_accel],
            dtype=np.float64,
        )
        acceleration = np.clip(
            desired_acceleration, lower_acceleration, upper_acceleration
        )
        next_velocity = velocity + acceleration * cfg.dt
        return {
            "acceleration_body_mps2": {
                "x": float(acceleration[0]),
                "y": float(acceleration[1]),
                "z": float(acceleration[2]),
            },
            "predicted_velocity_body_mps": {
                "x": float(next_velocity[0]),
                "y": float(next_velocity[1]),
                "z": float(next_velocity[2]),
            },
            "predicted_peak_abs_velocity_mps": {
                "x": float(max(abs(velocity[0]), abs(next_velocity[0]))),
                "y": float(max(abs(velocity[1]), abs(next_velocity[1]))),
                "z": float(max(abs(velocity[2]), abs(next_velocity[2]))),
            },
            "mode": "overspeed_brake",
            "speed_limit_mps": {
                "x": float(speed_limits[0]),
                "y": float(speed_limits[1]),
                "z": float(speed_limits[2]),
            },
            "target_speed_mps": cfg.target_speed,
            "progress_rate_mps": 0.0,
            "delta_s_m": 0.0,
            "contour_error_m": 0.0,
            "lag_error_m": 0.0,
            "objective": None,
            "path_tangent_body": {
                "x": float(tangent[0]),
                "y": float(tangent[1]),
                "z": float(tangent[2]),
            },
        }
