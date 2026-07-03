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
    # Lateral-velocity damping: bank to brake real sideways drift (m/s) measured
    # from the drone's velocity, not just the gate's image motion. A far gate
    # barely shifts in the image even while the drone slides sideways, so the
    # image-only D term misses the drift until it's too late. This term reacts to
    # the measured lateral speed directly so the drone arrests drift early.
    # Sign mirrors roll_gain_deg (negative): +lateral speed -> opposite bank.
    roll_velocity_gain_deg: float = -3.0
    max_roll_velocity_correction_deg: float = 6.0
    # Bank toward the gate frame tilt seen in the image (orientation_deg).
    orientation_roll_gain_deg: float = 0.65
    max_orientation_roll_deg: float = 12.0
    # Gentle yaw to help square the gate frame when it is visibly rotated.
    orientation_yaw_gain_deg: float = 0.35
    max_orientation_yaw_deg: float = 6.0
    max_roll_near_deg: float = 8.0
    max_roll_far_deg: float = 6.0
    # size_ratio (gate width as a fraction of the image) at/above which the gate
    # is treated as "close" (gentle corrections) and at/below which it is "far"
    # (aggressive corrections). Everything in between is interpolated.
    size_close: float = 0.15
    size_far: float = 0.50
    # Correction multiplier applied when the gate is as far away as size_far. A
    # distant gate barely moves in the image even when the drone is well off the
    # line, so scale its proportional correction up by this factor to bank early.
    schedule_far_max: float = 5.0
    # The thrust (vertical) axis has far more lag/momentum than roll, so scaling
    # it by the full far-gate factor above makes the drone porpoise (oscillate up
    # and down). Cap the vertical correction's distance scaling well below the
    # roll scaling to keep altitude tracking smooth.
    thrust_schedule_far_max: float = 1.0
    # Single cruise pitch for the whole flight (negative = slight nose-down).
    # Kept constant whether or not a gate is visible so pitch never jumps on
    # detection. Less negative than -0.3 = slower, stable forward speed.
    
    # SPEED CONTROL
    cruise_forward_pitch_deg: float = -0.0001 #control the speed of the drone
    pitch_smooth_alpha: float = 0.006 # avoid jerky movements
    # UP CONTROL
    thrust_gain: float = 0.15
    thrust_rate_gain: float = -0.10
    max_thrust_correction: float = 0.30
    # Extra vertical authority when the gate sits low in the frame (drone too high).
    low_in_frame_threshold: float = 0.25
    low_in_frame_extra_gain: float = 0.40

    # --- Early descend cue for far + low gates ---
    # The main vertical PD is intentionally capped at thrust_schedule_far_max for
    # a FAR gate so it doesn't porpoise. The side effect is that a gate sitting
    # low (and often only glimpsed near the bottom of the frame) while still far
    # away barely produces any drop, so the drone arrives too high and dives late.
    # This adds a separate descend bias that engages ONLY when a gate is both far
    # (so the normal PD is throttled) and clearly low in the frame. When fully
    # engaged it commands enough descent to cut thrust to MIN_FLIGHT_THRUST_DESCENT
    # (see controller.py's descent-aware floor), so the drone drops hard and early
    # toward a low far gate without affecting near-gate behavior.
    # It scans every reported gate candidate (not just the primary one) so a far
    # low gate caught only in a corner still triggers the early drop.
    early_descend_enabled: bool = True
    # Only engage for gates farther than this (meters). Inside this range the
    # normal vertical PD already has full authority, so no early bias is needed.
    early_descend_distance_min_m: float = 180.0
    # Gate must be at least this far below the image center (offset_y, +down) to
    # count as "low". Below this it's treated as roughly centered vertically.
    early_descend_offset_y_min: float = 0.30
    # Descend bias per unit of (offset_y - threshold). Steep enough that a clearly
    # low far gate ramps the bias up to early_descend_max within ~0.3 of offset_y,
    # so the drone commits to a real drop rather than a token nudge.
    early_descend_gain: float = 1.0
    # Hard cap on the early descend bias. Set to the controller's full descent
    # authority (max_thrust_correction) so a fully-engaged early drop commands a
    # descent of ~0.30, which is past controller.py's DESCENT_FLOOR_FULL (0.28).
    # That relaxes the thrust floor all the way to MIN_FLIGHT_THRUST_DESCENT (0.0),
    # i.e. the drone fully cuts thrust and drops, instead of only easing down.
    early_descend_max: float = 0.25
    # Ignore very weak candidate detections so render/JPEG noise in a corner
    # doesn't trigger a phantom descent.
    early_descend_confidence_min: float = 0.35

    # --- Near-gate climb taper (anti-jump on pass-through) ---
    # Right as the drone passes through a gate, the gate slides up/out of frame
    # so offset_y swings negative and its rate spikes, AND the inner-aperture
    # centering can shove hard vertically. Both command a CLIMB at the exact
    # moment the drone is already entering the opening, so it "jumps" up and
    # exits too high -- which then leaves it above a low/far final gate and
    # unable to see it. This tapers ONLY the positive (climb) thrust correction
    # as the gate gets close (large size_ratio), so the drone coasts through at
    # its current height instead of ballooning up. Descent authority is left
    # fully intact (so it can still drop toward a low gate), and far gates are
    # untouched (taper = 1.0) so normal tracking / early-descend still work.
    near_gate_climb_taper_enabled: bool = True
    # size_ratio at which the climb taper starts easing the climb down.
    near_gate_climb_taper_size_start: float = 0.35
    # size_ratio at/above which the climb is fully suppressed (min scale).
    near_gate_climb_taper_size_end: float = 0.55
    # Remaining fraction of the climb correction when right at the gate. 0.0 =
    # no climb allowed on final approach; raise toward 1.0 to allow more climb.
    near_gate_climb_min_scale: float = 0.0

    roll_smooth_alpha: float = 0.15
    # EMA factor for the offset derivative (D term). The raw frame-to-frame rate
    # is noisy (vision jitter), so it is low-pass filtered: lower = smoother but
    # laggier damping, higher = snappier but noisier. Applied only on new frames.
    offset_rate_filter_alpha: float = 0.4
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

    # --- Inner-frame centering / corner avoidance ---
    # Once the inner gate aperture is detected (the drone is close enough that
    # the inner corners are in view), actively steer so the image-center white
    # cross stays in the MIDDLE of the inner frame, away from the corners,
    # instead of just freezing the roll once the cross slips inside. The signals
    # (inner_recenter_x/y) are normalized by the aperture half-size, so 0 = white
    # cross dead-center in the opening and +/-1 = white cross on an inner edge.
    inner_avoid_enabled: bool = True
    # Engage only once the inner aperture fills at least this fraction of the
    # frame, so a tiny/noisy far-away inner detection doesn't yank the drone.
    inner_engage_size_ratio: float = 0.18
    inner_roll_gain_deg: float = -6.0          # roll per unit aperture-offset_x
    inner_thrust_gain: float = 0.18            # thrust per unit aperture-offset_y
    inner_max_roll_deg: float = 6.0
    inner_max_thrust_correction: float = 0.25
    # Stronger shove as the white cross drifts toward a corner/edge: corrections
    # are scaled by (1 + corner_boost * proximity), proximity in [0, 1] where 1
    # means the cross is right on an inner edge.
    inner_corner_boost: float = 20.0
    # Small deadband around the aperture center so the drone holds steady (no
    # jitter) once the white cross is comfortably in the middle of the opening.
    inner_center_deadband: float = 25.10
    # Inner-corner alignment deadband. The per-axis inner_center_deadband above
    # only zeroes ONE axis at a time, so the boosted corner correction keeps
    # chasing the other axis and can fling the white cross from one inner corner
    # into another (the drone "hits the corners"). This is the inner-aperture
    # analogue of the outer align_* deadbands below: once the cross is within
    # this normalized radius of the aperture center on BOTH axes, declare the
    # drone "inner aligned" and stop commanding roll/yaw + hold thrust steady so
    # it coasts straight through the middle of the opening instead of correcting
    # into a corner. Normalized by the aperture half-size (0 = dead-center,
    # 1 = on an inner edge); keep it a bit larger than inner_center_deadband so
    # there is a clear settled zone.
    inner_align_recenter_deadband: float = 0.18
    # Alignment deadband: once the gate's center white cross sits under the
    # image-center white cross (|offset_x| small) AND the frame is square
    # (|orientation| small), stop commanding roll/yaw so the drone flies
    # straight through instead of endlessly micro-rotating. These match the
    # overlay's "ALIGNED" thresholds in vision_rx.py. (offset_y is corrected by
    # thrust, not roll/yaw, so it isn't part of the rotational deadband.)
    align_offset_x_deadband: float = 0.12
    align_orientation_deadband_deg: float = 6.0


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
        # Derivative (D-term) state. The vision detection only refreshes at camera
        # frame rate (~30 Hz) while update() runs at the control rate (~250 Hz), so
        # the offset rate must be measured between genuinely NEW frames (tracked by
        # frame_id) and held in between, not recomputed every tick (which made it
        # spike for one tick on a new frame and read 0.0 the rest of the time).
        self._prev_frame_id = None
        self._prev_frame_time = None
        self._offset_x_rate = 0.0
        self._offset_y_rate = 0.0
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
        self._prev_frame_id = None
        self._prev_frame_time = None
        self._offset_x_rate = 0.0
        self._offset_y_rate = 0.0
        self._tracking = False
        self._reacquire_until = 0.0
        self._straighten_until = 0.0

    def notify_gate_passed(self):
        now = time.time()
        self._straighten_until = now + self._cfg.post_pass_straighten_s
        self._tracking = False
        self._prev_offset_x = None
        self._prev_offset_y = None
        self._prev_frame_id = None
        self._prev_frame_time = None
        self._offset_x_rate = 0.0
        self._offset_y_rate = 0.0
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
        # this is showing how big the gate looks and how close it is to the drone
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

    def _early_descend_bias(self, detection, primary_offset_y, primary_distance_m):
        # Gentle, capped descend bias (<= 0) for a gate that is BOTH far and low.
        # Scans the primary detection plus every reported gate candidate so a far
        # low gate seen only in a corner still drives an early drop. Returns 0.0
        # when nothing qualifies.
        cfg = self._cfg
        if not cfg.early_descend_enabled:
            return 0.0

        worst_low = 0.0  # most-below-center offset_y among qualifying far gates

        def _consider(distance_m, offset_y, confidence):
            nonlocal worst_low
            if confidence is not None and confidence < cfg.early_descend_confidence_min:
                return
            # Unknown distance is treated as "far" (a tiny/uncertain blob), which
            # is exactly the glimpsed far gate we want to react to.
            if distance_m is not None and distance_m < cfg.early_descend_distance_min_m:
                return
            if offset_y > worst_low:
                worst_low = offset_y

        _consider(primary_distance_m, primary_offset_y, None)
        for cand in (detection.get("gate_candidates") or []):
            _consider(
                cand.get("distance_m"),
                float(cand.get("offset_y", 0.0)),
                float(cand.get("confidence", 0.0)),
            )

        if worst_low <= cfg.early_descend_offset_y_min:
            return 0.0

        bias = cfg.early_descend_gain * (worst_low - cfg.early_descend_offset_y_min)
        return -_clamp(bias, 0.0, cfg.early_descend_max)

    def update(self, detection, current_velocity_x=0.0, lateral_velocity=0.0):
        cfg = self._cfg
        now = time.time()
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
            self._prev_frame_id = None
            self._prev_frame_time = None
            self._offset_x_rate = 0.0
            self._offset_y_rate = 0.0
            self._roll_deg *= (1.0 - cfg.roll_smooth_alpha)
            self._yaw_deg *= (1.0 - cfg.roll_smooth_alpha)
            # No gate in view: level off the forward pitch so the drone stops
            # moving forward and holds position, while leaving thrust (altitude)
            # untouched (thrust_correction stays neutral, so the controller's
            # base + altitude-hold thrust keeps the drone at its current height).
            return GateCommand(
                roll_deg=self._roll_deg,
                pitch_deg=self._smooth_pitch_toward(
                    0.0, alpha=cfg.straighten_smooth_alpha
                ),
                yaw_deg=self._yaw_deg,
                thrust_correction=0.0,
                tracking=False,
                straightening=False,
            )

        offset_x, offset_y, orientation_deg, size_ratio = self._tracking_signals(detection)

        # Finite-difference rates for the D term (damps overshoot/oscillation).
        # The detection only updates at camera frame rate while this runs at the
        # control rate, so measure the rate ONLY when a genuinely new frame
        # arrives (new frame_id) using the real time since the last frame, then
        # low-pass filter it and HOLD it between frames. Recomputing every tick
        # made the rate read 0.0 most ticks and spike on the one tick a new frame
        # landed (with a ~4 ms control dt), so the damping was a periodic jolt
        # instead of a steady velocity term.
        frame_id = detection.get("frame_id")
        new_frame = self._prev_frame_id is None or frame_id != self._prev_frame_id
        if new_frame:
            if (
                self._prev_offset_x is not None
                and self._prev_offset_y is not None
                and self._prev_frame_time is not None
            ):
                fdt = max(1e-3, now - self._prev_frame_time)
                raw_x_rate = (offset_x - self._prev_offset_x) / fdt
                raw_y_rate = (offset_y - self._prev_offset_y) / fdt
                alpha = cfg.offset_rate_filter_alpha
                self._offset_x_rate += alpha * (raw_x_rate - self._offset_x_rate)
                self._offset_y_rate += alpha * (raw_y_rate - self._offset_y_rate)
            self._prev_offset_x = offset_x
            self._prev_offset_y = offset_y
            self._prev_frame_time = now
            self._prev_frame_id = frame_id
        offset_x_rate = self._offset_x_rate
        offset_y_rate = self._offset_y_rate

        if not self._tracking:
            self._reacquire_until = now + cfg.reacquire_ramp_s
        self._tracking = True

        schedule = self._distance_schedule(size_ratio)
        if now < self._reacquire_until:
            ramp = 1.0 - (self._reacquire_until - now) / cfg.reacquire_ramp_s
            schedule *= max(0.0, ramp)

        # Are the two white crosses lined up (gate center under image center) and
        # is the frame square? If so, stop rotating: command wings-level roll and
        # straight-ahead yaw so the drone holds its heading through the gate.
        aligned = (
            abs(offset_x) <= cfg.align_offset_x_deadband
            and abs(orientation_deg) <= cfg.align_orientation_deadband_deg
        )

        # Inner-frame corner avoidance: once the inner aperture is detected and
        # big enough in the frame, keep the white cross in the MIDDLE of the
        # opening (away from the corners) by steering on the aperture-normalized
        # offset, with a stronger shove the closer the cross drifts to an edge.
        inner_rx = detection.get("inner_recenter_x")
        inner_ry = detection.get("inner_recenter_y")
        inner_active = (
            cfg.inner_avoid_enabled
            and bool(detection.get("inner_detected"))
            and inner_rx is not None
            and inner_ry is not None
            and float(detection.get("inner_size_ratio", 0.0)) >= cfg.inner_engage_size_ratio
        )
        inner_corner_boost = 1.0
        # True once the white cross is comfortably inside the inner aperture on
        # BOTH axes -> stop steering and coast straight through the middle so the
        # boosted corner correction can't push the drone into an inner corner.
        inner_aligned = False
        if inner_active:
            inner_rx = float(inner_rx)
            inner_ry = float(inner_ry)
            proximity = min(1.0, max(abs(inner_rx), abs(inner_ry)))
            inner_corner_boost = 1.0 + cfg.inner_corner_boost * proximity
            inner_aligned = (
                abs(inner_rx) <= cfg.inner_align_recenter_deadband
                and abs(inner_ry) <= cfg.inner_align_recenter_deadband
            )

        # Distance-dependent bank limit, needed by every branch below so the
        # added drift-damping shares the same physical bank ceiling.
        roll_limit = self._roll_limit_deg(size_ratio)

        # Lateral-velocity damping (computed once, applied in every branch). This
        # banks against the drone's measured sideways speed so drift is arrested
        # even when the gate looks centered (small offset_x) or the drone is
        # close. It is intentionally NOT scaled by the distance schedule: it is a
        # real m/s velocity, so it must bite early while the gate is still far.
        lateral_drift_roll = _clamp(
            cfg.roll_velocity_gain_deg * lateral_velocity,
            -cfg.max_roll_velocity_correction_deg,
            cfg.max_roll_velocity_correction_deg,
        )
        # Give the damping its own guaranteed authority: the near-gate roll limit
        # can be tuned very small, which would otherwise swallow the correction
        # exactly when it matters (close to the gate). The total bank is capped
        # by whichever is larger so the base steering tuning is preserved.
        total_roll_limit = max(roll_limit, cfg.max_roll_velocity_correction_deg)

        if inner_active:
            if inner_aligned:
                # White cross is settled in the middle of the opening: stop
                # rolling so the corner-boosted correction can't overshoot the
                # cross into an inner corner. Fly straight through.
                roll_target = 0.0
            else:
                # Park the white cross at the aperture center: drive recenter_x -> 0.
                # Deadband keeps the drone steady once it is safely centered.
                rx = inner_rx if abs(inner_rx) > cfg.inner_center_deadband else 0.0
                roll_target = _clamp(
                    cfg.inner_roll_gain_deg * rx * inner_corner_boost,
                    -cfg.inner_max_roll_deg,
                    cfg.inner_max_roll_deg,
                )
            yaw_target = 0.0
        elif aligned:
            roll_target = 0.0
            yaw_target = 0.0
        else:
            # Lateral (roll) PD, scaled up for distant gates and clamped to a
            # distance-dependent bank limit.
            roll_pd = (
                cfg.roll_gain_deg * offset_x
                + cfg.roll_rate_gain_deg * offset_x_rate
            )
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
            yaw_target = _clamp(
                cfg.orientation_yaw_gain_deg * orientation_deg * schedule,
                -cfg.max_orientation_yaw_deg,
                cfg.max_orientation_yaw_deg,
            )

        roll_target = _clamp(
            roll_target + lateral_drift_roll,
            -total_roll_limit,
            total_roll_limit,
        )

        self._roll_deg += cfg.roll_smooth_alpha * (roll_target - self._roll_deg)
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

        if inner_active:
            if inner_aligned:
                # Cross is settled vertically in the opening: hold thrust steady
                # (no vertical shove) so it can't be driven into a top/bottom
                # inner corner while passing through.
                thrust_correction = 0.0
            else:
                # Keep the white cross vertically centered in the opening too, so it
                # stays clear of the top/bottom inner corners. Drive recenter_y -> 0
                # (positive = aperture below the cross -> ease off thrust to descend).
                ry = inner_ry if abs(inner_ry) > cfg.inner_center_deadband else 0.0
                thrust_correction = _clamp(
                    -cfg.inner_thrust_gain * ry * inner_corner_boost,
                    -cfg.inner_max_thrust_correction,
                    cfg.inner_max_thrust_correction,
                )
        else:
            # Far + low gate: nudge the drone down early so it isn't still too
            # high when the gate finally grows close. Skipped when inner_active
            # (that path only fires up close, where the normal PD already drops).
            early_descend = self._early_descend_bias(
                detection,
                primary_offset_y=offset_y,
                primary_distance_m=detection.get("distance_estimate_m"),
            )
            if early_descend < 0.0:
                thrust_correction = _clamp(
                    thrust_correction + early_descend,
                    -cfg.max_thrust_correction,
                    cfg.max_thrust_correction,
                )

        # Near-gate climb taper: as the gate gets close (about to be passed),
        # scale down ONLY a positive (climb) correction so the drone doesn't
        # jump up while passing through and exit too high. Descent is untouched.
        if cfg.near_gate_climb_taper_enabled and thrust_correction > 0.0:
            span = (
                cfg.near_gate_climb_taper_size_end
                - cfg.near_gate_climb_taper_size_start
            )
            if span > 0.0:
                t = _clamp(
                    (size_ratio - cfg.near_gate_climb_taper_size_start) / span,
                    0.0,
                    1.0,
                )
            else:
                t = 1.0 if size_ratio >= cfg.near_gate_climb_taper_size_start else 0.0
            climb_scale = _lerp(1.0, cfg.near_gate_climb_min_scale, t)
            thrust_correction *= climb_scale

        return GateCommand(
            roll_deg=self._roll_deg,
            pitch_deg=self._smooth_pitch_toward(self._cruise_pitch_deg()),
            yaw_deg=self._yaw_deg,
            thrust_correction=thrust_correction,
            tracking=True,
            straightening=False,
        )
