import time
from dataclasses import dataclass

# =============================================================================
# TUNING GUIDE — gate_controller.py
# =============================================================================
# Every value in GateControlConfig below is LIVE and is edited ONLY in this file.
#
# Do NOT duplicate or override these in controller.py.
#
# For baseline flight (cruise throttle, altitude hold, idle roll/yaw), see
# controller.py → section "BASELINE FLIGHT SETTINGS".
# =============================================================================


def _clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def _lerp(a, b, t):
    return a + (b - a) * t


@dataclass
class GateControlConfig:
    """Gate-following tuning panel — edit values in THIS file only (@tune).

    Each field below is tagged  @tune: gate_controller.py  and is used directly
    at runtime. controller.py does not override any of them.
    """

    # =====================================================================
    # @tune: gate_controller.py — FORWARD SPEED
    # ---------------------------------------------------------------------
    # Single cruise pitch held for the whole flight (negative = nose-down =
    # faster). Kept constant whether or not a gate is visible so pitch never
    # jumps on detection. Near 0 = very slow/stable; more negative = faster.
    cruise_forward_pitch_deg: float = -0.0001  # @tune: gate_controller.py
    # How quickly pitch eases toward the target (smaller = smoother/less jerky).
    pitch_smooth_alpha: float = 0.006  # @tune: gate_controller.py

    # =====================================================================
    # @tune: gate_controller.py — LEFT / RIGHT STEERING (roll PID)
    # ---------------------------------------------------------------------
    roll_gain_deg: float = -8.0          # @tune: gate_controller.py — P
    roll_rate_gain_deg: float = -5.0     # @tune: gate_controller.py — D
    # I: accumulates lateral offset so a small but persistent off-centre bias
    # (which pure PD leaves as steady-state error) is slowly driven out. Reset
    # on gate loss/pass so it can't wind up.
    roll_integral_gain_deg: float = -1.0  # @tune: gate_controller.py — I
    # Anti-windup: hard cap on how many degrees of roll the I term may add.
    max_roll_integral_deg: float = 0.5  # @tune: gate_controller.py
    # How quickly the commanded roll eases toward the target (output smoothing).
    roll_smooth_alpha: float = 0.15  # @tune: gate_controller.py

    # =====================================================================
    # @tune: gate_controller.py — GATE FRAME ALIGNMENT
    # ---------------------------------------------------------------------
    # Bank toward the gate frame tilt seen in the image (orientation_deg).
    orientation_roll_gain_deg: float = 0.2  # @tune: gate_controller.py
    max_orientation_roll_deg: float = 0.4  # @tune: gate_controller.py
    # Gentle yaw to help square up a visibly rotated gate frame.
    orientation_yaw_gain_deg: float = 0.2  # @tune: gate_controller.py
    max_orientation_yaw_deg: float = 0.10  # @tune: gate_controller.py

    # =====================================================================
    # @tune: gate_controller.py — UP / DOWN (vertical thrust PID)
    # ---------------------------------------------------------------------
    
    # P: proportional gain -> gate above or below center climb or descend now
    thrust_gain: float = 0.40            # @tune: gate_controller.py — P
    # D: rate gain -> move forward center fast and ease off reduce bounce
    thrust_rate_gain: float = -0.35      # @tune: gate_controller.py — D
    # I -> if gate off center add a slow extra climb descend until that bias goes away
    thrust_integral_gain: float = 0.05  # @tune: gate_controller.py — I
    
    # clamp (limit) for the integral ±0.12 
    max_thrust_integral: float = 0.12  # @tune: gate_controller.py
    # P + D + I is limit at ±0.30
    max_thrust_correction: float = 0.10  # @tune: gate_controller.py
    # How low does the gate have to be in the camera before I start reacting more aggressively?
    low_in_frame_threshold: float = 0.30  # @tune: gate_controller.py
    # decides how much to increase the gain
    low_in_frame_extra_gain: float = 0.10 # @tune: gate_controller.py

    # =====================================================================
    # @tune: gate_controller.py — DISTANCE SCALING
    # ---------------------------------------------------------------------
    # size_ratio (gate width as a fraction of the image) at/above which the gate
    # is treated as "close" (gentle corrections) and at/below which it is "far"
    # (aggressive corrections). Everything in between is interpolated.
    size_close: float = 0.5  # @tune: gate_controller.py
    size_far: float = 0.12  # @tune: gate_controller.py
    # Correction multiplier applied when the gate is as far away as size_far. A
    # distant gate barely moves in the image even when the drone is well off the
    # line, so scale its proportional correction up by this factor to bank early.
    schedule_far_max: float = 3.0  # @tune: gate_controller.py
    # The vertical axis has far more lag than roll, so the full far-gate factor
    # makes it porpoise (up/down wave). Cap its distance scaling well below roll.
    thrust_schedule_far_max: float = 1.0  # @tune: gate_controller.py
    # Max bank allowed when the gate is close vs far (far allows a steeper bank
    # so the aggressive far-gate correction isn't clamped back to the near limit).
    max_roll_near_deg: float = 0.5  # @tune: gate_controller.py
    max_roll_far_deg: float = 6.0  # @tune: gate_controller.py

    # =====================================================================
    # @tune: gate_controller.py — DETECTION VALIDITY
    # ---------------------------------------------------------------------
    confidence_min: float = 0.45  # @tune: gate_controller.py
    detection_max_age_s: float = 0.60  # @tune: gate_controller.py

    # =====================================================================
    # @tune: gate_controller.py — GATE LOSS / REACQUIRE / POST-PASS STRAIGHTEN
    # ---------------------------------------------------------------------
    # On gate loss, ease corrections back in over this long instead of snapping.
    reacquire_ramp_s: float = 0.25  # @tune: gate_controller.py
    # After the sim reports a gate pass, fly flat (no bank/tilt) for this long
    # unless a new ahead gate is detected (0 = don't straighten).
    post_pass_straighten_s: float = 0.0  # @tune: gate_controller.py
    straighten_smooth_alpha: float = 0.12  # @tune: gate_controller.py
    straighten_pitch_deg: float = -0.01  # @tune: gate_controller.py
    # Ignore tiny low detections right after a pass (usually the gate just flown
    # through) so they don't cancel the straighten.
    straighten_trailing_size_max: float = 0.22  # @tune: gate_controller.py
    straighten_trailing_offset_y_min: float = 0.35  # @tune: gate_controller.py


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
        # Accumulated error for the I term (offset * seconds), per axis.
        self._integral_offset_x = 0.0
        self._integral_offset_y = 0.0

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
        self._integral_offset_x = 0.0
        self._integral_offset_y = 0.0

    def notify_gate_passed(self):
        now = time.time()
        self._straighten_until = now + self._cfg.post_pass_straighten_s
        self._tracking = False
        self._prev_offset_x = None
        self._prev_offset_y = None
        self._reacquire_until = now + self._cfg.reacquire_ramp_s
        # Drop the accumulated error so the next gate starts integrating fresh.
        self._integral_offset_x = 0.0
        self._integral_offset_y = 0.0

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
            # No valid measurement to integrate - clear the accumulator so it
            # doesn't wind up while the gate is out of view.
            self._integral_offset_x = 0.0
            self._integral_offset_y = 0.0
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

        # Integral (I) term: accumulate offset * dt for each axis, clamping the
        # accumulator so the resulting contribution can never exceed the
        # configured degree/thrust limits (anti-windup). Gains of 0 disable it.
        if cfg.roll_integral_gain_deg != 0.0:
            roll_i_state_limit = cfg.max_roll_integral_deg / abs(cfg.roll_integral_gain_deg)
            self._integral_offset_x = _clamp(
                self._integral_offset_x + offset_x * dt,
                -roll_i_state_limit,
                roll_i_state_limit,
            )
        else:
            self._integral_offset_x = 0.0
        if cfg.thrust_integral_gain != 0.0:
            thrust_i_state_limit = cfg.max_thrust_integral / abs(cfg.thrust_integral_gain)
            self._integral_offset_y = _clamp(
                self._integral_offset_y + offset_y * dt,
                -thrust_i_state_limit,
                thrust_i_state_limit,
            )
        else:
            self._integral_offset_y = 0.0
        roll_integral = cfg.roll_integral_gain_deg * self._integral_offset_x
        thrust_integral = -cfg.thrust_integral_gain * self._integral_offset_y

        schedule = self._distance_schedule(size_ratio)
        if now < self._reacquire_until:
            ramp = 1.0 - (self._reacquire_until - now) / cfg.reacquire_ramp_s
            schedule *= max(0.0, ramp)

        # Lateral (roll) PID: P+D scaled up for distant gates, plus the I term.
        # Only P/D are distance-scaled - the integral is already a slow, bounded
        # bias correction, so scaling it too would invite windup on far gates.
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
            roll_pd * schedule + roll_integral + orientation_roll,
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

        # Vertical (thrust) PID. Uses a gentler distance scaling than roll: the
        # vertical axis is laggy, so the full far-gate schedule would make it
        # overshoot and porpoise (up/down wave). The reacquire ramp still applies
        # because it can only pull the schedule below this cap, never above it.
        # The I term is added unscaled (same reasoning as roll) and the final
        # sum is clamped to the overall thrust-correction limit.
        thrust_schedule = min(schedule, cfg.thrust_schedule_far_max)
        vertical_gain = cfg.thrust_gain
        if offset_y > cfg.low_in_frame_threshold:
            vertical_gain += cfg.low_in_frame_extra_gain
        thrust_pd = (
            -vertical_gain * offset_y
            + cfg.thrust_rate_gain * offset_y_rate
        )
        thrust_correction = _clamp(
            thrust_pd * thrust_schedule + thrust_integral,
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
