# WASD + altitude control via camera-relative velocity

import math
import keyboard
import controller

# Minimal navigation constants
SPEED_LATERAL = 2.0      # m/s for forward/back/strafe
SPEED_VERTICAL = 2.0     # m/s for camera-up/down
CAMERA_PITCH_OFFSET_DEG = 20.0  # camera is mounted +20 degrees pitch relative to body

MANUAL_FORWARD_PITCH_DEG = 5.0  # nose-down angle for forward motion
MANUAL_BACK_PITCH_DEG = -5.0    # nose-up angle for backward motion
MANUAL_THRUST_STEP = 0.02       # thrust adjustment for E/C input
MANUAL_BASE_THRUST = 0.2


def _get_pressed_keys():
    try:
        return {
            'w': keyboard.is_pressed('w'),
            'a': keyboard.is_pressed('a'),
            's': keyboard.is_pressed('s'),
            'd': keyboard.is_pressed('d'),
            'e': keyboard.is_pressed('e'),
            'c': keyboard.is_pressed('c'),
        }
    except Exception:
        return {k: False for k in ('w', 'a', 's', 'd', 'e', 'c')}


def _resolve_conflicts(keys):
    if keys['w'] and keys['s']:
        keys['w'] = keys['s'] = False
    if keys['a'] and keys['d']:
        keys['a'] = keys['d'] = False
    if keys['e'] and keys['c']:
        keys['e'] = keys['c'] = False
    return keys


def _calculate_manual_attitude_command(keys):
    pitch_deg = 0.0
    if keys['w']:
        pitch_deg = MANUAL_FORWARD_PITCH_DEG
    elif keys['s']:
        pitch_deg = MANUAL_BACK_PITCH_DEG

    thrust = MANUAL_BASE_THRUST
    if keys['e']:
        thrust += MANUAL_THRUST_STEP
    elif keys['c']:
        thrust -= MANUAL_THRUST_STEP

    thrust = min(
        max(thrust, controller.MIN_FLIGHT_THRUST),
        controller.MAX_FLIGHT_THRUST,
    )

    return pitch_deg, thrust


def handle_user_input(armed_controller):
    """Minimal handler that maps WASD+E/C into attitude+thrust commands.

    - Reads keys and resolves conflicts.
    - Builds a small pitch command for forward/back movement.
    - Adjusts thrust for camera-up/camera-down.
    - Preserves current yaw heading.
    """
    if not getattr(armed_controller, 'sim_conn', None):
        return
    if getattr(armed_controller, 'system_boot_ms', None) is None:
        raise ValueError('armed_controller.system_boot_ms is required')

    keys = _resolve_conflicts(_get_pressed_keys())
    pitch_deg, thrust = _calculate_manual_attitude_command(keys)

    body_att = getattr(armed_controller, 'data', {}).get('attitude', {}) or {}
    body_yaw = float(body_att.get('yaw', 0.0))

    controller.update_attitude_flight_control(
        armed_controller.sim_conn,
        armed_controller.system_boot_ms,
        thrust=thrust,
        roll_deg=0.0,
        pitch_deg=pitch_deg,
        yaw_deg=body_yaw,
    )
    