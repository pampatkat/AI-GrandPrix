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
    # Bank toward the gate frame tilt seen in the image (orientation_deg).
    orientation_roll_gain_deg: float = 0.65
    max_orientation_roll_deg: float = 12.0
    # Gentle yaw to help square the gate frame when it is visibly rotated.
    orientation_yaw_gain_deg: float = 0.35
    max_orientation_yaw_deg: float = 6.0
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
    schedule_far_max: float = 3.0
    # The thrust (vertical) axis has far more lag/momentum than roll, so scaling
    # it by the full far-gate factor above makes the drone porpoise (oscillate up
    # and down). Cap the vertical correction's distance scaling well below the
    # roll scaling to keep altitude tracking smooth.
    thrust_schedule_far_max: float = 1.0
    # Single cruise pitch for the whole flight (negative = slight nose-down).
    # Kept constant whether or not a gate is visible so pitch never jumps on
    # detection. Less negative than -0.3 = slower, stable forward speed.
    cruise_forward_pitch_deg: float = -0.12
    pitch_smooth_alpha: float = 0.06
    thrust_gain: float = 0.10
    thrust_rate_gain: float = -0.10
    max_thrust_correction: float = 0.15
    # Extra vertical authority when the gate sits low in the frame (drone too high).
    low_in_frame_threshold: float = 0.25
    low_in_frame_extra_gain: float = 0.40
    roll_smooth_alpha: float = 0.15
    reacquire_ramp_s: float = 0.25
    # After the simulator reports a gate pass, fly flat (no bank/tilt) for this long
    # unless a new ahead gate is detected.
    post_pass_straighten_s: float = 1.5
    straighten_smooth_alpha: float = 0.12
    straighten_pitch_deg: float = 0.0
    # Ignore tiny low detections right after a pass (usually the gate just flown through).
    straighten_trailing_size_max: float = 0.22
    straighten_trailing_offset_y_min: float = 0.35
    confidence_min: float = 0.45
    detection_max_age_s: float = 0.35


@dataclass
class GateCommand:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    thrust_correction: float
    tracking: bool
    straightening: bool = False


class GateController:
    def __init__(self, config: GateControlConfig):
        self._cfg = config
        self._roll_deg = 0.0
        self._pitch_deg = config.cruise_forward_pitch_deg
        self._yaw_deg = 0.0
        self._prev_offset_x = None
        self._prev_offset_y = None
        self._prev_update_time = None
        self._tracking = False
        self._reacquire_until = 0.0
        self._straighten_until = 0.0

    def reset(self):
        self._roll_deg = 0.0
        self._pitch_deg = self._cfg.cruise_forward_pitch_deg
        self._yaw_deg = 0.0
        self._prev_offset_x = None
        self._prev_offset_y = None
        self._prev_update_time = None
        self._tracking = False
        self._reacquire_until = 0.0
        self._straighten_until = 0.0

    def notify_gate_passed(self):
        now = time.time()
        self._straighten_until = now + self._cfg.post_pass_straighten_s
        self._tracking = False
        self._prev_offset_x = None
        self._prev_offset_y = None
        self._reacquire_until = now + self._cfg.reacquire_ramp_s

    def _cancel_straighten(self):
        self._straighten_until = 0.0

    def _in_post_pass_straighten(self, now):
        return now < self._straighten_until

    def _is_trailing_gate_detection(self, detection):
        size_ratio = float(detection.get('size_ratio', 0.0))
        offset_y = float(detection.get('offset_y', 0.0))
        cfg = self._cfg
        return (
            size_ratio < cfg.straighten_trailing_size_max
            and offset_y > cfg.straighten_trailing_offset_y_min
        )

    def _should_resume_tracking_from_straighten(self, detection):
        if not self._is_valid_detection(detection):
            return False
        if self._is_trailing_gate_detection(detection):
            return False
        return True

    def _smooth_pitch_toward(self, target_deg, alpha=None):
        if alpha is None:
            alpha = self._cfg.pitch_smooth_alpha
        self._pitch_deg += alpha * (target_deg - self._pitch_deg)
        return self._pitch_deg

    def _straighten_command(self, cfg):
        alpha = cfg.straighten_smooth_alpha
        self._roll_deg += alpha * (0.0 - self._roll_deg)
        self._yaw_deg += alpha * (0.0 - self._yaw_deg)
        return GateCommand(
            roll_deg=self._roll_deg,
            pitch_deg=self._smooth_pitch_toward(cfg.straighten_pitch_deg),
            yaw_deg=self._yaw_deg,
            thrust_correction=0.0,
            tracking=False,
            straightening=True,
        )

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

    def _tracking_signals(self, detection):
        """Prefer PnP pose when valid; fall back to 2D image heuristics."""
        if detection.get('pnp_valid'):
            return (
                float(detection.get('pnp_offset_x', 0.0)),
                float(detection.get('pnp_offset_y', 0.0)),
                float(detection.get('pnp_orientation_deg', 0.0)),
                float(detection.get('pnp_size_ratio', detection.get('size_ratio', 0.2))),
            )
        return (
            float(detection.get('offset_x', 0.0)),
            float(detection.get('offset_y', 0.0)),
            float(detection.get('orientation_deg', 0.0)),
            float(detection.get('size_ratio', 0.2)),
        )

    def _cruise_pitch_deg(self):
        return self._cfg.cruise_forward_pitch_deg

    def update(self, detection, current_velocity_x=0.0):
        cfg = self._cfg
        now = time.time()
        dt = 0.004
        if self._prev_update_time is not None:
            dt = max(0.001, now - self._prev_update_time)
        self._prev_update_time = now

        if self._in_post_pass_straighten(now):
            if self._should_resume_tracking_from_straighten(detection):
                self._cancel_straighten()
            else:
                return self._straighten_command(cfg)

        if not self._is_valid_detection(detection):
            # Lost the gate: start a re-acquire ramp so corrections ease back in
            # rather than snapping, and let the held roll decay toward level.
            if self._tracking:
                self._reacquire_until = now + cfg.reacquire_ramp_s
            self._tracking = False
            self._prev_offset_x = None
            self._prev_offset_y = None
            self._roll_deg *= (1.0 - cfg.roll_smooth_alpha)
            self._yaw_deg *= (1.0 - cfg.roll_smooth_alpha)
            return GateCommand(
                roll_deg=self._roll_deg,
                pitch_deg=self._smooth_pitch_toward(self._cruise_pitch_deg()),
                yaw_deg=self._yaw_deg,
                thrust_correction=0.0,
                tracking=False,
                straightening=False,
            )

        offset_x, offset_y, orientation_deg, size_ratio = self._tracking_signals(detection)

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
        orientation_roll = _clamp(
            cfg.orientation_roll_gain_deg * orientation_deg * schedule,
            -cfg.max_orientation_roll_deg,
            cfg.max_orientation_roll_deg,
        )
        roll_target = _clamp(
            roll_pd * schedule + orientation_roll,
            -roll_limit,
            roll_limit,
        )
        self._roll_deg += cfg.roll_smooth_alpha * (roll_target - self._roll_deg)

        yaw_target = _clamp(
            cfg.orientation_yaw_gain_deg * orientation_deg * schedule,
            -cfg.max_orientation_yaw_deg,
            cfg.max_orientation_yaw_deg,
        )
        self._yaw_deg += cfg.roll_smooth_alpha * (yaw_target - self._yaw_deg)

        # Vertical (thrust) PD. Uses a gentler distance scaling than roll: the
        # vertical axis is laggy, so the full far-gate schedule would make it
        # overshoot and porpoise (up/down wave). The reacquire ramp still applies
        # because it can only pull the schedule below this cap, never above it.
        thrust_schedule = min(schedule, cfg.thrust_schedule_far_max)
        vertical_gain = cfg.thrust_gain
        if offset_y > cfg.low_in_frame_threshold:
            vertical_gain += cfg.low_in_frame_extra_gain
        thrust_pd = (
            -vertical_gain * offset_y
            + cfg.thrust_rate_gain * offset_y_rate
        )
        thrust_correction = _clamp(
            thrust_pd * thrust_schedule,
            -cfg.max_thrust_correction,
            cfg.max_thrust_correction,
        )

        return GateCommand(
            roll_deg=self._roll_deg,
            pitch_deg=self._smooth_pitch_toward(self._cruise_pitch_deg()),
            yaw_deg=self._yaw_deg,
            thrust_correction=thrust_correction,
            tracking=True,
            straightening=False,
        )
