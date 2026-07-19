"""Thread-safe timestamped IMU history used for delayed camera-state replay."""

import threading
from collections import deque

import numpy as np


class IMUHistory:
    """Keep a short IMU window and estimate the stationary accelerometer bias."""

    def __init__(self, max_age_s=2.0, max_samples=2000):
        self.max_age_s = float(max_age_s)
        self.samples = deque(maxlen=int(max_samples))
        self.lock = threading.Lock()
        self.calibrating = True
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.bias_samples = 0
        self.latest_source = "unavailable"

    def start_calibration(self):
        with self.lock:
            self.calibrating = True
            self.accel_bias[:] = 0.0
            self.bias_samples = 0

    def freeze_calibration(self):
        with self.lock:
            self.calibrating = False

    def add(self, timestamp_s, acceleration, gyro, source="sensor"):
        timestamp_s = float(timestamp_s)
        acceleration = np.asarray(acceleration, dtype=np.float64).reshape(3)
        gyro = np.asarray(gyro, dtype=np.float64).reshape(3)
        if not (np.all(np.isfinite(acceleration)) and np.all(np.isfinite(gyro))):
            return

        with self.lock:
            if self.calibrating:
                self.bias_samples += 1
                alpha = 1.0 / min(self.bias_samples, 500)
                self.accel_bias += alpha * (acceleration - self.accel_bias)
            self.samples.append(
                {
                    "timestamp_s": timestamp_s,
                    "acceleration_raw": acceleration.copy(),
                    "gyro": gyro.copy(),
                    "source": str(source),
                }
            )
            self.latest_source = str(source)
            cutoff = timestamp_s - self.max_age_s
            while self.samples and self.samples[0]["timestamp_s"] < cutoff:
                self.samples.popleft()

    def window(self, start_s, end_s, include_previous=True):
        """Return bias-corrected samples in chronological order for replay."""
        start_s = float(start_s)
        end_s = float(end_s)
        with self.lock:
            rows = list(self.samples)
            bias = self.accel_bias.copy()

        selected = []
        previous = None
        for sample in rows:
            timestamp_s = sample["timestamp_s"]
            if timestamp_s <= start_s:
                previous = sample
            if start_s < timestamp_s <= end_s:
                selected.append(sample)
        if include_previous and previous is not None:
            selected.insert(0, previous)

        return [
            {
                "timestamp_s": sample["timestamp_s"],
                "acceleration": sample["acceleration_raw"] - bias,
                "gyro": sample["gyro"].copy(),
                "source": sample["source"],
            }
            for sample in selected
        ]

    def stats(self, now_s=None):
        with self.lock:
            rows = list(self.samples)
            bias = self.accel_bias.copy()
            calibrating = self.calibrating
            bias_samples = self.bias_samples
            source = self.latest_source
        if not rows:
            return {
                "samples": 0,
                "rate_hz": None,
                "age_s": None,
                "calibrating": calibrating,
                "bias_samples": bias_samples,
                "source": source,
                "accel_bias": bias.tolist(),
            }
        if now_s is None:
            now_s = rows[-1]["timestamp_s"]
        span = rows[-1]["timestamp_s"] - rows[0]["timestamp_s"]
        rate = (len(rows) - 1) / span if span > 1e-6 else None
        return {
            "samples": len(rows),
            "rate_hz": rate,
            "age_s": max(0.0, float(now_s) - rows[-1]["timestamp_s"]),
            "calibrating": calibrating,
            "bias_samples": bias_samples,
            "source": source,
            "accel_bias": bias.tolist(),
        }
