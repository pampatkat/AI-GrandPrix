# WASD + altitude control via camera-relative velocity

import math
import keyboard
import controller

# Minimal navigation constants
SPEED_LATERAL = 2.0      # m/s for forward/back/strafe
SPEED_VERTICAL = 2.0     # m/s for camera-up/down
CAMERA_PITCH_OFFSET_DEG = 20.0  # camera is mounted +20 degrees pitch relative to body


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


def _calculate_camera_velocity(keys):
    vx = (SPEED_LATERAL if keys['w'] else 0.0) + (-SPEED_LATERAL if keys['s'] else 0.0)
    vy = (SPEED_LATERAL if keys['d'] else 0.0) + (-SPEED_LATERAL if keys['a'] else 0.0)
    vz = (SPEED_VERTICAL if keys['c'] else 0.0) + (-SPEED_VERTICAL if keys['e'] else 0.0)
    return vx, vy, vz


def handle_user_input(armed_controller):
    """Minimal handler that maps WASD+E/C into camera-relative velocity.

    - Reads keys and resolves conflicts.
    - Builds camera-frame velocity vector.
    - Transforms that velocity to NED using the camera attitude.
    - Sends the resulting zero or non-zero velocity command.
    """
    if not getattr(armed_controller, 'sim_conn', None):
        return

    keys = _resolve_conflicts(_get_pressed_keys())
    vx_cam, vy_cam, vz_cam = _calculate_camera_velocity(keys)

    # Always send an explicit velocity command, even when all inputs are zero.
    body_att = getattr(armed_controller, 'data', {}).get('attitude', {}) or {}
    body_roll = float(body_att.get('roll', 0.0))
    body_pitch = float(body_att.get('pitch', 0.0))
    body_yaw = float(body_att.get('yaw', 0.0))

    camera_pitch = body_pitch + math.radians(CAMERA_PITCH_OFFSET_DEG)
    camera_roll = body_roll
    camera_yaw = body_yaw

    cr = math.cos(camera_yaw)
    sr = math.sin(camera_yaw)
    ct = math.cos(camera_pitch)
    st = math.sin(camera_pitch)
    cp = math.cos(camera_roll)
    sp = math.sin(camera_roll)

    R00 = cr * ct
    R01 = cr * st * sp - sr * cp
    R02 = cr * st * cp + sr * sp

    R10 = sr * ct
    R11 = sr * st * sp + cr * cp
    R12 = sr * st * cp - cr * sp

    R20 = -st
    R21 = ct * sp
    R22 = ct * cp

    vx_ned = R00 * vx_cam + R01 * vy_cam + R02 * vz_cam
    vy_ned = R10 * vx_cam + R11 * vy_cam + R12 * vz_cam
    vz_ned = R20 * vx_cam + R21 * vy_cam + R22 * vz_cam

    controller.update_position_flight_control(
        armed_controller.sim_conn,
        armed_controller.system_boot_ms,
        {'vx': vx_ned, 'vy': vy_ned, 'vz': vz_ned}
    )
    