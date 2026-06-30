# WASD + yaw/thrust control via body attitude setpoints

import math
import keyboard
import controller

# Minimal navigation constants
SPEED_LATERAL = 2.0      # m/s for forward/back/strafe
SPEED_VERTICAL = 2.0     # m/s for camera-up/down
CAMERA_PITCH_OFFSET_DEG = 20.0  # camera is mounted +20 degrees pitch relative to body

MANUAL_FORWARD_PITCH_DEG =  5.0     # nose-down angle for forward motion
MANUAL_BACK_PITCH_DEG =     -5.0    # nose-up angle for backward motion
MANUAL_LEFT_ROLL_DEG =      5.0     # roll-left angle for leftward motion
MANUAL_RIGHT_ROLL_DEG =     -5.0    # roll-right angle for rightward motion
MANUAL_YAW_STEP_DEG =       5.0     # heading step for Q/E yaw control
MANUAL_THRUST_STEP =        0.1     # thrust adjustment for V/C input
MANUAL_BASE_THRUST =        0.275


def _get_pressed_keys():
    try:
        return {
            'w': keyboard.is_pressed('w'),
            'a': keyboard.is_pressed('a'),
            's': keyboard.is_pressed('s'),
            'd': keyboard.is_pressed('d'),
            'q': keyboard.is_pressed('q'),
            'e': keyboard.is_pressed('e'),
            'v': keyboard.is_pressed('v'),
            'c': keyboard.is_pressed('c'),
        }
    except Exception:
        return {k: False for k in ('w', 'a', 's', 'd', 'q', 'e', 'v', 'c')}


def _resolve_conflicts(keys):
    if keys['w'] and keys['s']:
        keys['w'] = keys['s'] = False
    if keys['a'] and keys['d']:
        keys['a'] = keys['d'] = False
    if keys['q'] and keys['e']:
        keys['q'] = keys['e'] = False
    if keys['v'] and keys['c']:
        keys['v'] = keys['c'] = False
    return keys


def _calculate_manual_attitude_command(keys):
    roll_deg = 0.0
    if keys['a']:
        roll_deg = MANUAL_LEFT_ROLL_DEG
    elif keys['d']:
        roll_deg = MANUAL_RIGHT_ROLL_DEG

    pitch_deg = 0.0
    if keys['w']:
        pitch_deg = MANUAL_FORWARD_PITCH_DEG
    elif keys['s']:
        pitch_deg = MANUAL_BACK_PITCH_DEG

    yaw_delta_deg = 0.0
    if keys['q']:
        yaw_delta_deg = MANUAL_YAW_STEP_DEG
    elif keys['e']:
        yaw_delta_deg = -MANUAL_YAW_STEP_DEG

    thrust = MANUAL_BASE_THRUST
    if keys['v']:
        thrust += MANUAL_THRUST_STEP
    elif keys['c']:
        thrust -= MANUAL_THRUST_STEP

    thrust = min(
        max(thrust, controller.MIN_FLIGHT_THRUST),
        controller.MAX_FLIGHT_THRUST,
    )

    return roll_deg, pitch_deg, yaw_delta_deg, thrust


def handle_user_input(armed_controller):
    """Minimal handler that maps WASD+Q/E/V/C into attitude+thrust commands.

    - Reads keys and resolves conflicts.
    - Builds a small pitch command for forward/back movement.
    - Adjusts yaw heading from Q/E and thrust from V/C.
    - Preserves current heading while applying commanded attitude.
    """
    if not getattr(armed_controller, 'sim_conn', None):
        return
    if getattr(armed_controller, 'system_boot_ms', None) is None:
        raise ValueError('armed_controller.system_boot_ms is required')

    keys = _resolve_conflicts(_get_pressed_keys())
    roll_deg, pitch_deg, yaw_delta_deg, thrust = _calculate_manual_attitude_command(keys)

    body_att = getattr(armed_controller, 'data', {}).get('attitude', {}) or {}
    body_yaw_rad = float(body_att.get('yaw', 0.0))
    body_yaw_deg = math.degrees(body_yaw_rad)
    yaw_deg = body_yaw_deg + yaw_delta_deg

    controller.update_attitude_flight_control(
        armed_controller.sim_conn,
        armed_controller.system_boot_ms,
        thrust=thrust,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )
    