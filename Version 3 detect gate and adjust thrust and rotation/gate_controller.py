import time
from dataclasses import dataclass


def _clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def _lerp(a, b, t):
    return a + (b - a) * t


@dataclass
class GateControlConfig:
    roll_gain_deg: float = -10.0
    roll_rate_gain_deg: float = -2.0
    max_roll_near_deg: float = 8.0
    max_roll_far_deg: float = 18.0
    # size_ratio (gate width as a fraction of the image) at/above which the gate
    # is treated as "close" (gentle corrections) and at/below which it is "far"
    # (aggressive corrections). Everything in between is interpolated.
    size_close: float = 0.5
    size_far: float = 0.12
    # Correction multiplier applied when the gate is as far away as size_far. A
    # distant gate barely moves in the image even when the drone is well off the
    # line, so scale its proportional correction up by this factor to bank early.
    schedule_far_max: float = 10.0
    # The thrust (vertical) axis has far more lag/momentum than roll, so scaling
    # it by the full far-gate factor above makes the drone porpoise (oscillate up
    # and down). Cap the vertical correction's distance scaling well below the
    # roll scaling to keep altitude tracking smooth.
    thrust_schedule_far_max: float = 1.0
    base_forward_pitch_deg: float = -0.3
    thrust_gain: float = 0.10
    thrust_rate_gain: float = -0.10
    max_thrust_correction: float = 0.15
    roll_smooth_alpha: float = 0.15
    reacquire_ramp_s: float = 0.25
    confidence_min: float = 0.45
    detection_max_age_s: float = 0.35


@dataclass
class GateCommand:
    roll_deg: float
    pitch_deg: float
    thrust_correction: float
    tracking: bool


class GateController:
    def __init__(self, config: GateControlConfig):
        self._cfg = config
        self._roll_deg = 0.0
        self._prev_offset_x = None
        self._prev_offset_y = None
        self._prev_update_time = None
        self._tracking = False
        self._reacquire_until = 0.0

    def reset(self):
        self._roll_deg = 0.0
        self._prev_offset_x = None
        self._prev_offset_y = None
        self._prev_update_time = None
        self._tracking = False
        self._reacquire_until = 0.0

    def _distance_schedule(self, size_ratio):
        # 1.0 when the gate is close, ramping up to schedule_far_max as it gets
        # far. Multiplies the image-space corrections so a small/distant gate
        # produces a much larger bank/climb than its tiny pixel offset would.
        cfg = self._cfg
        if size_ratio >= cfg.size_close:
            return 1.0
        span = cfg.size_close - cfg.size_far
        if span <= 0.0:
            return cfg.schedule_far_max
        t = _clamp((cfg.size_close - size_ratio) / span, 0.0, 1.0)
        return 1.0 + t * (cfg.schedule_far_max - 1.0)

    def _roll_limit_deg(self, size_ratio):
        # Allow a steeper bank when the gate is far so the aggressive far-gate
        # correction isn't immediately clamped back to the gentle near limit.
        cfg = self._cfg
        if size_ratio >= cfg.size_close:
            return cfg.max_roll_near_deg
        span = cfg.size_close - cfg.size_far
        if span <= 0.0:
            return cfg.max_roll_far_deg
        t = _clamp((cfg.size_close - size_ratio) / span, 0.0, 1.0)
        return _lerp(cfg.max_roll_near_deg, cfg.max_roll_far_deg, t)

    def _is_valid_detection(self, detection):
        if not detection or not detection.get('detected'):
            return False
        if detection.get('confidence', 0.0) < self._cfg.confidence_min:
            return False
        if time.time() - detection.get('time', 0.0) > self._cfg.detection_max_age_s:
            return False
        return True

    def update(self, detection, current_velocity_x=0.0):
        cfg = self._cfg
        now = time.time()
        dt = 0.004
        if self._prev_update_time is not None:
            dt = max(0.001, now - self._prev_update_time)
        self._prev_update_time = now

        if not self._is_valid_detection(detection):
            # Lost the gate: start a re-acquire ramp so corrections ease back in
            # rather than snapping, and let the held roll decay toward level.
            if self._tracking:
                self._reacquire_until = now + cfg.reacquire_ramp_s
            self._tracking = False
            self._prev_offset_x = None
            self._prev_offset_y = None
            self._roll_deg *= (1.0 - cfg.roll_smooth_alpha)
            return GateCommand(
                roll_deg=self._roll_deg,
                pitch_deg=cfg.base_forward_pitch_deg,
                thrust_correction=0.0,
                tracking=False,
            )

        offset_x = float(detection.get('offset_x', 0.0))
        offset_y = float(detection.get('offset_y', 0.0))
        size_ratio = float(detection.get('size_ratio', 0.2))

        # Finite-difference rates for the D term (damps overshoot/oscillation).
        offset_x_rate = 0.0
        offset_y_rate = 0.0
        if self._prev_offset_x is not None:
            offset_x_rate = (offset_x - self._prev_offset_x) / dt
        if self._prev_offset_y is not None:
            offset_y_rate = (offset_y - self._prev_offset_y) / dt
        self._prev_offset_x = offset_x
        self._prev_offset_y = offset_y

        if not self._tracking:
            self._reacquire_until = now + cfg.reacquire_ramp_s
        self._tracking = True

        schedule = self._distance_schedule(size_ratio)
        if now < self._reacquire_until:
            ramp = 1.0 - (self._reacquire_until - now) / cfg.reacquire_ramp_s
            schedule *= max(0.0, ramp)

        # Lateral (roll) PD, scaled up for distant gates and clamped to a
        # distance-dependent bank limit.
        roll_pd = (
            cfg.roll_gain_deg * offset_x
            + cfg.roll_rate_gain_deg * offset_x_rate
        )
        roll_limit = self._roll_limit_deg(size_ratio)
        roll_target = _clamp(roll_pd * schedule, -roll_limit, roll_limit)
        self._roll_deg += cfg.roll_smooth_alpha * (roll_target - self._roll_deg)

        # Vertical (thrust) PD. Uses a gentler distance scaling than roll: the
        # vertical axis is laggy, so the full far-gate schedule would make it
        # overshoot and porpoise (up/down wave). The reacquire ramp still applies
        # because it can only pull the schedule below this cap, never above it.
        thrust_schedule = min(schedule, cfg.thrust_schedule_far_max)
        thrust_pd = (
            -cfg.thrust_gain * offset_y
            + cfg.thrust_rate_gain * offset_y_rate
        )
        thrust_correction = _clamp(
            thrust_pd * thrust_schedule,
            -cfg.max_thrust_correction,
            cfg.max_thrust_correction,
        )

        return GateCommand(
            roll_deg=self._roll_deg,
            pitch_deg=cfg.base_forward_pitch_deg,
            thrust_correction=thrust_correction,
            tracking=True,
        )
