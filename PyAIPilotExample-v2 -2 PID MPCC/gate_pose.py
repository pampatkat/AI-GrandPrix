"""Metric gate pose estimation from YOLO corner keypoints."""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for the simulator's 640 x 360 FPV stream."""

    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 180.0
    calibration_width: int = 640
    calibration_height: int = 360

    def matrix(self, image_width=640, image_height=360):
        """Return K, scaling the supplied calibration only for resized frames."""
        scale_x = float(image_width) / float(self.calibration_width)
        scale_y = float(image_height) / float(self.calibration_height)
        return np.array(
            [
                [self.fx * scale_x, 0.0, self.cx * scale_x],
                [0.0, self.fy * scale_y, self.cy * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def camera_to_body_rotation(pitch_offset_deg=20.0):
    """Return the specified camera-to-body pitch correction matrix R_B^C."""
    theta = math.radians(float(pitch_offset_deg))
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )


def opencv_to_body_axis_camera(vector):
    """Map OpenCV [right, down, forward] to camera [forward, right, down]."""
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([vector[2], vector[0], vector[1]], dtype=np.float64)


def process_pnp_to_body(tvec_opencv, pitch_offset_deg=20.0):
    """Permute an OpenCV PnP translation, then correct the camera pitch."""
    camera_aerospace = opencv_to_body_axis_camera(tvec_opencv)
    return camera_to_body_rotation(pitch_offset_deg) @ camera_aerospace


class GatePoseEstimator:
    """Use the four inner gate corners and ``cv2.solvePnP`` to recover pose."""

    CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")

    def __init__(
        self,
        gate_width_m=1.5,
        gate_height_m=1.5,
        intrinsics=None,
        camera_pitch_deg=20.0,
        min_keypoint_confidence=0.20,
        max_reprojection_error_px=12.0,
        min_depth_m=0.15,
        max_depth_m=80.0,
    ):
        if gate_width_m <= 0.0 or gate_height_m <= 0.0:
            raise ValueError("Gate dimensions must be positive")
        self.gate_width_m = float(gate_width_m)
        self.gate_height_m = float(gate_height_m)
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.rotation_body_from_camera = camera_to_body_rotation(camera_pitch_deg)

        half_width = self.gate_width_m * 0.5
        half_height = self.gate_height_m * 0.5
        # Gate-local convention: X right, Y up, with clockwise correspondence
        # TL, TR, BR, BL. The 1.5 m default therefore lies at +/- 0.75 m.
        self.object_points = np.array(
            [
                [-half_width, half_height, 0.0],
                [half_width, half_height, 0.0],
                [half_width, -half_height, 0.0],
                [-half_width, -half_height, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def invalid(reason):
        return {
            "valid": False,
            "reason": str(reason),
            "opencv_camera_position_m": None,
            "camera_position_m": None,
            "body_position_m": None,
            "distance_m": None,
            "bearing_body_deg": None,
            "gate_orientation_body_deg": None,
            "reprojection_error_px": None,
        }

    def estimate(self, keypoints, image_width=640, image_height=360):
        """Estimate the gate-center pose from named YOLO keypoint dictionaries."""
        point_map = {point.get("name"): point for point in (keypoints or [])}
        if not all(name in point_map for name in self.CORNER_ORDER):
            return self.invalid("four inner corners are required")

        image_points = []
        for name in self.CORNER_ORDER:
            point = point_map[name]
            confidence = point.get("confidence")
            if confidence is not None and confidence < self.min_keypoint_confidence:
                return self.invalid(f"low-confidence corner: {name}")
            x = float(point["x"])
            y = float(point["y"])
            if not (math.isfinite(x) and math.isfinite(y)):
                return self.invalid(f"non-finite corner: {name}")
            image_points.append((x, y))

        image_points = np.asarray(image_points, dtype=np.float64)
        camera_matrix = self.intrinsics.matrix(image_width, image_height)
        distortion = np.zeros((5, 1), dtype=np.float64)

        candidates = []
        solve_errors = []
        for method in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
            try:
                success, candidate_rotation, candidate_translation = cv2.solvePnP(
                    self.object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=method,
                )
                if not success:
                    continue
                candidate_projected, _ = cv2.projectPoints(
                    self.object_points,
                    candidate_rotation,
                    candidate_translation,
                    camera_matrix,
                    distortion,
                )
                candidate_residuals = (
                    candidate_projected.reshape(4, 2) - image_points
                )
                candidate_error = float(
                    np.sqrt(
                        np.mean(
                            np.sum(candidate_residuals * candidate_residuals, axis=1)
                        )
                    )
                )
                candidates.append(
                    (
                        candidate_error,
                        candidate_rotation,
                        candidate_translation,
                    )
                )
            except cv2.error as exc:
                solve_errors.append(str(exc))

        if not candidates:
            if solve_errors:
                return self.invalid(f"solvePnP failed: {solve_errors[-1]}")
            return self.invalid("solvePnP did not converge")

        reprojection_error, rotation_vector, translation_vector = min(
            candidates, key=lambda candidate: candidate[0]
        )

        opencv_camera_position = translation_vector.reshape(3).astype(np.float64)
        depth = float(opencv_camera_position[2])
        if not self.min_depth_m <= depth <= self.max_depth_m:
            return self.invalid("estimated depth is outside the configured range")

        if reprojection_error > self.max_reprojection_error_px:
            return self.invalid("reprojection error is too large")

        object_rotation_camera, _ = cv2.Rodrigues(rotation_vector)
        camera_position = opencv_to_body_axis_camera(opencv_camera_position)
        opencv_to_body_axes = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        body_position = process_pnp_to_body(
            opencv_camera_position, self.camera_pitch_deg
        )
        object_rotation_body = (
            self.rotation_body_from_camera
            @ opencv_to_body_axes
            @ object_rotation_camera
        )
        gate_normal_body = object_rotation_body[:, 2]
        # A plane normal has two possible signs. Keep the diagnostic normal
        # pointing from the drone through the gate rather than back at it.
        if float(np.dot(gate_normal_body, body_position)) < 0.0:
            gate_normal_body = -gate_normal_body

        distance = float(np.linalg.norm(body_position))
        body_x, body_y, body_z = (float(value) for value in body_position)
        # Standard body axes: X forward, Y right, Z down.
        horizontal_bearing = math.degrees(math.atan2(body_y, body_x))
        vertical_bearing = math.degrees(
            math.atan2(body_z, math.hypot(body_x, body_y))
        )
        normal_yaw = math.degrees(
            math.atan2(float(gate_normal_body[1]), float(gate_normal_body[0]))
        )
        normal_pitch = math.degrees(
            math.atan2(
                float(gate_normal_body[2]),
                math.hypot(float(gate_normal_body[0]), float(gate_normal_body[1])),
            )
        )
        gate_roll = math.degrees(
            math.atan2(float(object_rotation_body[2, 0]), float(object_rotation_body[1, 0]))
        )

        return {
            "valid": True,
            "reason": None,
            "opencv_camera_position_m": {
                "right": float(opencv_camera_position[0]),
                "down": float(opencv_camera_position[1]),
                "forward": float(opencv_camera_position[2]),
            },
            "camera_position_m": {
                # P_C supplied to R_B^C: X forward, Y right, Z down.
                "x": float(camera_position[0]),
                "y": float(camera_position[1]),
                "z": float(camera_position[2]),
            },
            "body_position_m": {"x": body_x, "y": body_y, "z": body_z},
            "distance_m": distance,
            "bearing_body_deg": {
                "horizontal": horizontal_bearing,
                "vertical": vertical_bearing,
            },
            "gate_orientation_body_deg": {
                "yaw": normal_yaw,
                "pitch": normal_pitch,
                "roll": gate_roll,
            },
            "rotation_vector": [float(value) for value in rotation_vector.reshape(3)],
            "translation_vector_opencv": [
                float(value) for value in opencv_camera_position
            ],
            "reprojection_error_px": reprojection_error,
            "gate_dimensions_m": {
                "width": self.gate_width_m,
                "height": self.gate_height_m,
            },
            "camera_pitch_compensation_deg": self.camera_pitch_deg,
        }
