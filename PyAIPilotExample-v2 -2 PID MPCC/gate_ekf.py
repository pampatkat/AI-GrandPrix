"""Delayed-measurement EKF for relative gate pose with buffered IMU replay."""

import math
import time
from collections import deque

import numpy as np


class GatePoseEKF:
    """Track body-frame gate [position, velocity] and replay delayed vision."""

    def __init__(self, process_accel_std=3.0, measurement_std=0.12, max_coast_s=1.25):
        self.process_accel_std = float(process_accel_std)
        self.measurement_std = float(measurement_std)
        self.max_coast_s = float(max_coast_s)
        self.reset()

    def reset(self, gate_index=None):
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)
        self.initialized = False
        self.filter_time = None
        self.last_measurement_time = None
        self.gate_index = gate_index
        self.accepted_measurements = 0
        self.rejected_measurements = 0
        self.last_raw_position = None
        self.last_raw_measurement_time = None
        self.raw_visual_velocity = None
        self.raw_measurement_dt = None
        self.history = deque(maxlen=2000)
        self.last_vision_latency_s = None
        self.last_replayed_imu_samples = 0
        self.last_alignment_source = "uninitialized"
        self.last_imu_used = False

    def set_gate_index(self, gate_index):
        if gate_index is None:
            return
        gate_index = int(gate_index)
        if self.gate_index is None:
            self.gate_index = gate_index
        elif gate_index != self.gate_index:
            self.reset(gate_index)

    @staticmethod
    def _transition(dt):
        transition = np.eye(6, dtype=np.float64)
        transition[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
        return transition

    @staticmethod
    def _skew(vector):
        x, y, z = vector
        return np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )

    def _process_covariance(self, dt):
        identity = np.eye(3, dtype=np.float64)
        q = self.process_accel_std ** 2
        return q * np.block(
            [
                [identity * (dt ** 4 / 4.0), identity * (dt ** 3 / 2.0)],
                [identity * (dt ** 3 / 2.0), identity * (dt ** 2)],
            ]
        )

    def _save_snapshot(self):
        if not self.initialized or self.filter_time is None:
            return
        snapshot = (
            float(self.filter_time),
            self.state.copy(),
            self.covariance.copy(),
        )
        if self.history and abs(self.history[-1][0] - self.filter_time) < 1e-9:
            self.history[-1] = snapshot
        else:
            self.history.append(snapshot)

    def _predict_step(self, dt, acceleration=None, gyro=None):
        if dt <= 0.0:
            return
        acceleration = np.zeros(3) if acceleration is None else np.asarray(
            acceleration, dtype=np.float64
        ).reshape(3)
        gyro = np.zeros(3) if gyro is None else np.asarray(
            gyro, dtype=np.float64
        ).reshape(3)
        acceleration = np.clip(acceleration, -12.0, 12.0)
        gyro = np.clip(gyro, -8.0, 8.0)

        remaining = float(dt)
        while remaining > 1e-9:
            step = min(0.02, remaining)
            position = self.state[0:3].copy()
            velocity = self.state[3:6].copy()
            omega_cross = self._skew(gyro)
            relative_acceleration = -acceleration - omega_cross @ velocity
            position_rate = velocity - omega_cross @ position
            self.state[0:3] = (
                position + position_rate * step
                + 0.5 * relative_acceleration * step * step
            )
            self.state[3:6] = velocity + relative_acceleration * step

            rotation_jacobian = np.eye(3) - omega_cross * step
            transition = np.block(
                [
                    [rotation_jacobian, np.eye(3) * step],
                    [np.zeros((3, 3)), rotation_jacobian],
                ]
            )
            self.covariance = (
                transition @ self.covariance @ transition.T
                + self._process_covariance(step)
            )
            remaining -= step

    def predict(self, timestamp=None, imu_samples=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        if not self.initialized:
            self.filter_time = timestamp
            return 0
        if timestamp <= self.filter_time:
            return 0

        samples = sorted(
            (imu_samples or []), key=lambda sample: sample["timestamp_s"]
        )
        current_acceleration = None
        current_gyro = None
        used = 0
        for sample in samples:
            sample_time = float(sample["timestamp_s"])
            if sample_time <= self.filter_time:
                current_acceleration = sample.get("acceleration")
                current_gyro = sample.get("gyro")
                continue
            if sample_time > timestamp:
                break
            self._predict_step(
                sample_time - self.filter_time,
                current_acceleration,
                current_gyro,
            )
            self.filter_time = sample_time
            current_acceleration = sample.get("acceleration")
            current_gyro = sample.get("gyro")
            used += 1
            self._save_snapshot()

        self._predict_step(
            timestamp - self.filter_time,
            current_acceleration,
            current_gyro,
        )
        self.filter_time = timestamp
        self.last_imu_used = used > 0 or current_acceleration is not None
        self._save_snapshot()
        return used

    def _predict_from_history(self, timestamp, imu_history):
        samples = []
        if imu_history is not None and self.filter_time is not None:
            samples = imu_history.window(self.filter_time, timestamp)
        return self.predict(timestamp, samples)

    def _record_raw_measurement(self, measurement, timestamp):
        if self.last_raw_position is not None and self.last_raw_measurement_time is not None:
            raw_dt = timestamp - self.last_raw_measurement_time
            if 0.005 <= raw_dt <= 1.0:
                self.raw_visual_velocity = (
                    measurement - self.last_raw_position
                ) / raw_dt
                self.raw_measurement_dt = raw_dt
        self.last_raw_position = measurement.copy()
        self.last_raw_measurement_time = timestamp

    def _initialize(self, measurement, timestamp):
        self.state[0:3] = measurement
        self.state[3:6] = 0.0
        self.covariance = np.diag([0.04, 0.04, 0.09, 9.0, 9.0, 9.0])
        self.initialized = True
        self.filter_time = timestamp
        self.last_measurement_time = timestamp
        self.accepted_measurements = 1
        self._save_snapshot()

    def _correct(self, measurement, measurement_std):
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, 0:3] = np.eye(3, dtype=np.float64)
        std = self.measurement_std if measurement_std is None else float(measurement_std)
        measurement_covariance = np.eye(3, dtype=np.float64) * max(std, 1e-3) ** 2
        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_covariance
        )
        mahalanobis_squared = float(
            innovation.T @ np.linalg.solve(innovation_covariance, innovation)
        )
        if mahalanobis_squared > 16.27:
            if self.state[0] < 0.5 and measurement[0] > 1.5:
                return "new_gate"
            self.rejected_measurements += 1
            return False

        kalman_gain = (
            self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(6, dtype=np.float64)
        correction = identity - kalman_gain @ observation
        self.covariance = (
            correction @ self.covariance @ correction.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self.accepted_measurements += 1
        return True

    def observe(self, position, timestamp=None, gate_index=None, measurement_std=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        return self.observe_delayed(
            position,
            capture_timestamp=timestamp,
            present_timestamp=timestamp,
            imu_history=None,
            gate_index=gate_index,
            measurement_std=measurement_std,
            alignment_source="direct",
        )

    def observe_delayed(
        self,
        position,
        capture_timestamp,
        present_timestamp=None,
        imu_history=None,
        gate_index=None,
        measurement_std=None,
        alignment_source="camera_timestamp",
    ):
        capture_timestamp = float(capture_timestamp)
        present_timestamp = (
            time.time() if present_timestamp is None else float(present_timestamp)
        )
        present_timestamp = max(present_timestamp, capture_timestamp)
        self.set_gate_index(gate_index)
        measurement = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(measurement)):
            return False
        self._record_raw_measurement(measurement, capture_timestamp)

        if not self.initialized:
            self._initialize(measurement, capture_timestamp)
        else:
            if capture_timestamp < self.filter_time - 1e-6:
                snapshot = next(
                    (
                        item for item in reversed(self.history)
                        if item[0] <= capture_timestamp + 1e-9
                    ),
                    None,
                )
                if snapshot is None:
                    self.rejected_measurements += 1
                    self.last_alignment_source = "history_miss"
                    return False
                self.filter_time = snapshot[0]
                self.state = snapshot[1].copy()
                self.covariance = snapshot[2].copy()
                self.history = deque(
                    (item for item in self.history if item[0] <= self.filter_time),
                    maxlen=2000,
                )
            self._predict_from_history(capture_timestamp, imu_history)
            correction_result = self._correct(measurement, measurement_std)
            if correction_result == "new_gate":
                current_index = self.gate_index
                self.reset(current_index)
                self._record_raw_measurement(measurement, capture_timestamp)
                self._initialize(measurement, capture_timestamp)
            elif not correction_result:
                self._predict_from_history(present_timestamp, imu_history)
                return False
            self.last_measurement_time = capture_timestamp
            self._save_snapshot()

        self.last_vision_latency_s = present_timestamp - capture_timestamp
        self.last_alignment_source = str(alignment_source)
        self.last_replayed_imu_samples = self._predict_from_history(
            present_timestamp, imu_history
        )
        return True

    def estimate(self, timestamp=None, imu_history=None):
        timestamp = time.time() if timestamp is None else float(timestamp)
        self._predict_from_history(timestamp, imu_history)
        if not self.initialized or self.last_measurement_time is None:
            return {"valid": False, "source": "uninitialized"}

        measurement_age = max(0.0, timestamp - self.last_measurement_time)
        position = self.state[0:3]
        velocity = self.state[3:6]
        valid = measurement_age <= self.max_coast_s and float(position[0]) > -1.0
        distance = float(np.linalg.norm(position))
        source = "measured+imu_replay" if measurement_age <= 0.08 else "ekf_prediction"
        confidence = math.exp(-measurement_age / max(self.max_coast_s, 1e-6))
        return {
            "valid": bool(valid),
            "source": source,
            "gate_index": self.gate_index,
            "measurement_age_s": measurement_age,
            "vision_latency_ms": None
            if self.last_vision_latency_s is None
            else self.last_vision_latency_s * 1000.0,
            "replayed_imu_samples": self.last_replayed_imu_samples,
            "timestamp_alignment": self.last_alignment_source,
            "imu_propagation": self.last_imu_used,
            "confidence": confidence,
            "body_position_m": {
                "x": float(position[0]), "y": float(position[1]), "z": float(position[2])
            },
            "body_velocity_mps": {
                "x": float(velocity[0]), "y": float(velocity[1]), "z": float(velocity[2])
            },
            "ekf_drone_velocity_mps": {
                "x": float(-velocity[0]), "y": float(-velocity[1]), "z": float(-velocity[2])
            },
            "ekf_closing_speed_mps": float(-velocity[0]),
            "raw_visual_relative_velocity_mps": None
            if self.raw_visual_velocity is None
            else {
                "x": float(self.raw_visual_velocity[0]),
                "y": float(self.raw_visual_velocity[1]),
                "z": float(self.raw_visual_velocity[2]),
            },
            "raw_visual_closing_speed_mps": None
            if self.raw_visual_velocity is None
            else float(-self.raw_visual_velocity[0]),
            "raw_measurement_rate_hz": None
            if not self.raw_measurement_dt else float(1.0 / self.raw_measurement_dt),
            "distance_m": distance,
            "bearing_body_deg": {
                "horizontal": math.degrees(math.atan2(float(position[1]), float(position[0]))),
                "vertical": math.degrees(
                    math.atan2(float(position[2]), math.hypot(float(position[0]), float(position[1])))
                ),
            },
            "position_variance": [float(value) for value in np.diag(self.covariance)[0:3]],
            "accepted_measurements": self.accepted_measurements,
            "rejected_measurements": self.rejected_measurements,
        }
