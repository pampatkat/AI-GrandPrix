"""Focused tests for PnP pose, camera compensation, EKF coasting, and MPCC math."""

import math
import queue
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from gate_ekf import GatePoseEKF
from gate_detector import GateDetector
from fpv_display import FPVDisplay
from imu_buffer import IMUHistory
from mavlink_rx import MAVLinkRX
from gate_pose import (
    CameraIntrinsics,
    GatePoseEstimator,
    camera_to_body_rotation,
    opencv_to_body_axis_camera,
    process_pnp_to_body,
)
from mpcc import CasadiMPCC, MPCCConfig
from controller import (
    ACTIVE_FLIGHT_MIN_THRUST,
    Controller,
    FlightCommand,
    MPCCGateController,
    clamp_active_flight_thrust,
)


class GatePoseTests(unittest.TestCase):
    def test_inner_gate_object_points_are_exact_clockwise_square(self):
        estimator = GatePoseEstimator(gate_width_m=1.5, gate_height_m=1.5)
        np.testing.assert_allclose(
            estimator.object_points,
            np.array(
                [
                    [-0.75, 0.75, 0.0],
                    [0.75, 0.75, 0.0],
                    [0.75, -0.75, 0.0],
                    [-0.75, -0.75, 0.0],
                ]
            ),
        )

    def test_supplied_intrinsics_and_rotation(self):
        np.testing.assert_allclose(
            CameraIntrinsics().matrix(),
            np.array(
                [[320.0, 0.0, 320.0], [0.0, 320.0, 180.0], [0.0, 0.0, 1.0]]
            ),
        )
        theta = math.radians(20.0)
        expected = np.array(
            [
                [math.cos(theta), 0.0, math.sin(theta)],
                [0.0, 1.0, 0.0],
                [-math.sin(theta), 0.0, math.cos(theta)],
            ]
        )
        np.testing.assert_allclose(camera_to_body_rotation(), expected)
        raw_opencv = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            process_pnp_to_body(raw_opencv),
            expected @ np.array([4.0, 2.0, 3.0]),
        )

    def test_solve_pnp_recovers_synthetic_gate_center(self):
        estimator = GatePoseEstimator(max_reprojection_error_px=0.1)
        expected_camera = np.array([0.25, -0.12, 5.0], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            estimator.object_points,
            np.zeros(3, dtype=np.float64),
            expected_camera,
            estimator.intrinsics.matrix(),
            np.zeros(5, dtype=np.float64),
        )
        names = ("top_left", "top_right", "bottom_right", "bottom_left")
        keypoints = [
            {
                "name": name,
                "x": float(point[0]),
                "y": float(point[1]),
                "confidence": 1.0,
            }
            for name, point in zip(names, image_points.reshape(4, 2))
        ]
        pose = estimator.estimate(keypoints)
        self.assertTrue(pose["valid"], pose.get("reason"))
        recovered_opencv = np.array(
            list(pose["opencv_camera_position_m"].values())
        )
        np.testing.assert_allclose(recovered_opencv, expected_camera, atol=1e-5)
        expected_camera_axes = opencv_to_body_axis_camera(expected_camera)
        np.testing.assert_allclose(
            np.array(list(pose["camera_position_m"].values())),
            expected_camera_axes,
            atol=1e-5,
        )
        expected_body = camera_to_body_rotation() @ expected_camera_axes
        np.testing.assert_allclose(
            np.array(list(pose["body_position_m"].values())),
            expected_body,
            atol=1e-5,
        )

    def test_low_image_gate_maps_to_level_body_forward_axis(self):
        estimator = GatePoseEstimator(max_reprojection_error_px=0.1)
        expected_body = np.array([5.0, 0.0, 0.0], dtype=np.float64)
        camera_axes = camera_to_body_rotation().T @ expected_body
        opencv_translation = np.array(
            [camera_axes[1], camera_axes[2], camera_axes[0]], dtype=np.float64
        )
        # Gate-local Y is up while OpenCV camera Y is down, so a front-facing
        # gate carries a 180-degree rotation around its local X axis.
        front_facing_rotation, _ = cv2.Rodrigues(
            np.diag([1.0, -1.0, -1.0])
        )
        image_points, _ = cv2.projectPoints(
            estimator.object_points,
            front_facing_rotation,
            opencv_translation,
            estimator.intrinsics.matrix(),
            np.zeros(5, dtype=np.float64),
        )
        names = ("top_left", "top_right", "bottom_right", "bottom_left")
        keypoints = [
            {"name": name, "x": float(point[0]), "y": float(point[1]), "confidence": 1.0}
            for name, point in zip(names, image_points.reshape(4, 2))
        ]
        pose = estimator.estimate(keypoints)
        self.assertTrue(pose["valid"], pose.get("reason"))
        np.testing.assert_allclose(
            np.array(list(pose["body_position_m"].values())), expected_body, atol=1e-5
        )


class GateDetectorTests(unittest.TestCase):
    def test_predictive_roi_is_square_for_stable_cuda_shape(self):
        detector = GateDetector.__new__(GateDetector)
        detector.roi_enabled = True
        detector._last_bbox = [270.0, 110.0, 370.0, 250.0]
        detector._last_bbox_frame_id = 4
        detector._bbox_velocity = [0.0, 0.0, 0.0, 0.0]
        detector._roi_misses = 0
        detector.roi_scale = 1.8
        detector.image_size = 640
        detector.roi_image_size = 192
        image = np.zeros((360, 640, 3), dtype=np.uint8)

        crop, roi = detector._select_inference_region(image, frame_id=5)

        self.assertTrue(roi["used"])
        self.assertEqual(crop.shape[0], crop.shape[1])
        self.assertEqual(roi["x2"] - roi["x1"], roi["y2"] - roi["y1"])
        self.assertEqual(roi["input_size"], 192)

    def test_roi_results_are_restored_to_full_frame_pixels(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class FakeBoxes:
            xyxy = FakeTensor([[1.0, 2.0, 31.0, 42.0]])
            conf = FakeTensor([0.9])
            cls = FakeTensor([0.0])

        points = [
            [2.0, 3.0, 1.0], [30.0, 3.0, 1.0],
            [2.0, 40.0, 1.0], [30.0, 40.0, 1.0],
            [1.0, 2.0, 1.0], [31.0, 2.0, 1.0],
            [1.0, 42.0, 1.0], [31.0, 42.0, 1.0],
        ]
        fake_result = type("Result", (), {
            "boxes": FakeBoxes(),
            "keypoints": type("Keypoints", (), {"data": FakeTensor([points])})(),
        })()
        detector = GateDetector.__new__(GateDetector)
        detector.pose_estimator = GatePoseEstimator()
        detection = detector._serialize_result(
            fake_result, 640, 360, offset_x=100, offset_y=50
        )[0]
        self.assertEqual(detection["bbox"]["x1"], 101.0)
        self.assertEqual(detection["bbox"]["y1"], 52.0)
        self.assertEqual(detection["keypoints"][0]["x"], 102.0)
        self.assertEqual(detection["keypoints"][0]["y"], 53.0)

    def test_fpv_preview_throttles_without_copying_input_frame(self):
        display = FPVDisplay.__new__(FPVDisplay)
        display.max_fps = 10.0
        display._minimum_frame_interval = 0.1
        display._last_queued_time = 0.0
        display._frames = queue.Queue(maxsize=1)
        image = np.zeros((360, 640, 3), dtype=np.uint8)

        display.submit(1, image)
        frame_id, queued_image = display._frames.get_nowait()
        self.assertEqual(frame_id, 1)
        self.assertIs(queued_image, image)

        display.submit(2, image)
        self.assertTrue(display._frames.empty())


class GateEKFTests(unittest.TestCase):
    def test_filter_predicts_during_gate_loss_and_resets_for_next_gate(self):
        tracker = GatePoseEKF(max_coast_s=1.0)
        self.assertTrue(tracker.observe([5.0, 0.2, 0.1], timestamp=10.0, gate_index=2))
        self.assertTrue(tracker.observe([4.7, 0.18, 0.09], timestamp=10.1, gate_index=2))
        measured = tracker.estimate(10.1)
        coasted = tracker.estimate(10.3)
        self.assertTrue(coasted["valid"])
        self.assertEqual(coasted["source"], "ekf_prediction")
        self.assertLess(coasted["body_position_m"]["x"], measured["body_position_m"]["x"])
        self.assertAlmostEqual(measured["raw_visual_closing_speed_mps"], 3.0)
        self.assertGreater(measured["raw_measurement_rate_hz"], 9.9)

        tracker.set_gate_index(3)
        self.assertFalse(tracker.estimate(10.31)["valid"])

    def test_delayed_camera_correction_replays_imu_to_present(self):
        imu = IMUHistory(max_age_s=2.0)
        imu.freeze_calibration()
        for timestamp in (10.0, 10.025, 10.05, 10.075, 10.1):
            imu.add(timestamp, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        tracker = GatePoseEKF()
        tracker.observe([5.0, 0.0, 0.0], timestamp=10.0)
        tracker.estimate(10.1, imu)
        accepted = tracker.observe_delayed(
            [4.99875, 0.0, 0.0],
            capture_timestamp=10.05,
            present_timestamp=10.1,
            imu_history=imu,
            alignment_source="unit_test",
        )
        estimate = tracker.estimate(10.1, imu)
        self.assertTrue(accepted)
        self.assertAlmostEqual(estimate["vision_latency_ms"], 50.0, places=5)
        self.assertGreaterEqual(estimate["replayed_imu_samples"], 2)
        self.assertTrue(estimate["imu_propagation"])
        self.assertEqual(estimate["timestamp_alignment"], "unit_test")
        self.assertLess(estimate["body_position_m"]["x"], 4.99875)

    def test_imu_buffer_removes_stationary_bias(self):
        imu = IMUHistory(max_age_s=2.0)
        for index in range(20):
            imu.add(1.0 + index * 0.01, [0.2, -0.1, 9.7], [0.0, 0.0, 0.0])
        imu.freeze_calibration()
        samples = imu.window(1.1, 1.2)
        np.testing.assert_allclose(samples[-1]["acceleration"], [0.0, 0.0, 0.0], atol=1e-9)


class TimeSyncTests(unittest.TestCase):
    def test_standard_timesync_response_estimates_server_clock_offset(self):
        sent_ns = 1_700_000_000_000_000_000
        received_ns = sent_ns + 20_000_000
        server_midpoint_ns = sent_ns + 110_000_000
        data = {"timesync_last_request_ns": sent_ns}
        receiver = MAVLinkRX(None, data)
        response = SimpleNamespace(tc1=server_midpoint_ns, ts1=sent_ns)

        with mock.patch("mavlink_rx.time.time_ns", return_value=received_ns):
            receiver.on_timesync(response)

        self.assertEqual(
            data["timesync"]["server_to_local_offset_ns"], -100_000_000
        )
        self.assertAlmostEqual(data["timesync"]["round_trip_ms"], 20.0)


class MPCCTests(unittest.TestCase):
    def test_contour_and_lag_error_are_orthogonal(self):
        contour, lag = CasadiMPCC.contour_and_lag_error(
            position=[2.0, 3.0, 4.0], tangent=[0.0, 0.0, 1.0], progress=1.5
        )
        np.testing.assert_allclose(contour, [2.0, 3.0, 0.0])
        self.assertAlmostEqual(lag, 2.5)

    def test_solver_rewards_forward_progress(self):
        optimizer = CasadiMPCC(MPCCConfig(horizon=5, dt=0.1))
        if not optimizer.available:
            self.skipTest(optimizer.error)
        result = optimizer.solve([8.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertIsNotNone(result, optimizer.error)
        self.assertGreater(result["delta_s_m"], 0.0)
        self.assertGreaterEqual(result["acceleration_body_mps2"]["x"], 0.0)
        self.assertEqual(result["mode"], "optimized_speed_lock")
        for axis in ("x", "y", "z"):
            self.assertLessEqual(
                result["predicted_peak_abs_velocity_mps"][axis],
                result["speed_limit_mps"][axis] + 1e-5,
            )

    def test_overspeed_input_commands_braking(self):
        optimizer = CasadiMPCC(MPCCConfig(horizon=5, dt=0.1))
        if not optimizer.available:
            self.skipTest(optimizer.error)
        result = optimizer.solve([8.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        self.assertEqual(result["mode"], "overspeed_brake")
        self.assertLess(result["acceleration_body_mps2"]["x"], 0.0)
        self.assertLess(result["predicted_velocity_body_mps"]["x"], 2.0)

    def test_vertical_acceleration_uses_gate_level_and_vertical_velocity(self):
        config = MPCCConfig(horizon=5, dt=0.1)
        below = CasadiMPCC(config)
        above = CasadiMPCC(config)
        rising = CasadiMPCC(config)
        if not below.available:
            self.skipTest(below.error)

        below_result = below.solve([8.0, 0.0, 2.0], [0.0, 0.0, 0.0])
        above_result = above.solve([8.0, 0.0, -2.0], [0.0, 0.0, 0.0])
        rising_result = rising.solve([8.0, 0.0, 0.0], [0.0, 0.0, -0.8])

        self.assertGreater(below_result["acceleration_body_mps2"]["z"], 0.0)
        self.assertLess(above_result["acceleration_body_mps2"]["z"], 0.0)
        self.assertGreater(rising_result["acceleration_body_mps2"]["z"], 0.0)

    def test_roll_sign_and_center_aware_authority(self):
        right = MPCCGateController.roll_target_from_acceleration(1.0, 10.0)
        left = MPCCGateController.roll_target_from_acceleration(-1.0, -10.0)
        centered = MPCCGateController.roll_target_from_acceleration(2.0, 0.5)

        self.assertGreater(right, 0.0)
        self.assertLess(left, 0.0)
        self.assertLessEqual(
            abs(centered), MPCCGateController.CENTERED_ROLL_LIMIT_DEG
        )
        self.assertAlmostEqual(
            MPCCGateController.roll_limit_for_bearing(15.0),
            MPCCGateController.MAX_ROLL_DEG,
        )

    def test_pitch_has_bounded_forward_and_braking_authority(self):
        forward = MPCCGateController.pitch_target_from_acceleration(0.8)
        braking = MPCCGateController.pitch_target_from_acceleration(-1.0)
        excessive = MPCCGateController.pitch_target_from_acceleration(5.0)

        self.assertLess(forward, 0.0)
        self.assertGreater(braking, 0.0)
        self.assertLessEqual(abs(forward), MPCCGateController.MAX_PITCH_DEG)
        self.assertAlmostEqual(excessive, -MPCCGateController.MAX_PITCH_DEG)
        self.assertAlmostEqual(MPCCConfig().max_forward_accel, 0.8)
        self.assertAlmostEqual(MPCCConfig().max_forward_decel, 1.0)
        self.assertAlmostEqual(MPCCGateController.MAX_PITCH_DEG, 0.01)
        self.assertAlmostEqual(Controller.MAX_COMMAND_PITCH_DEG, 0.01)


class CollectiveSafetyTests(unittest.TestCase):
    @staticmethod
    def _controller():
        controller = Controller.__new__(Controller)
        controller.hover_thrust_estimate = 0.28
        controller.min_thrust = 0.25
        controller.max_thrust = 0.48
        controller.data = {}
        return controller

    def test_vertical_control_requires_a_fresh_camera_measurement(self):
        self.assertFalse(MPCCGateController.vertical_measurement_is_fresh(None))
        self.assertTrue(MPCCGateController.vertical_measurement_is_fresh(0.10))
        self.assertFalse(MPCCGateController.vertical_measurement_is_fresh(0.151))

    def test_default_gravity_calibration_is_not_the_actuator_floor(self):
        self.assertAlmostEqual(Controller.INITIAL_HOVER_THRUST, 0.28)
        self.assertAlmostEqual(Controller.MIN_THRUST, 0.25)
        self.assertAlmostEqual(ACTIVE_FLIGHT_MIN_THRUST, 0.25)
        self.assertAlmostEqual(clamp_active_flight_thrust(0.05), 0.25)
        self.assertAlmostEqual(clamp_active_flight_thrust(0.30), 0.30)

    def test_mpcc_down_acceleration_calculates_sub_hover_thrust(self):
        controller = self._controller()
        command = FlightCommand(
            tracking=True,
            gate_vertical_bearing_deg=10.0,
            gate_vertical_control_active=True,
            vertical_accel_down_mps2=2.0,
        )

        thrust = controller._calculate_collective(
            command, altitude_correction=0.0, altitude_observable=False
        )
        expected = max(
            0.25,
            0.28 * (1.0 - 2.0 / MPCCGateController.GRAVITY_MPS2),
        )
        self.assertAlmostEqual(thrust, expected)
        self.assertLess(thrust, 0.28)
        self.assertTrue(controller.data["collective_status"]["minimum_applied"])
        self.assertTrue(controller.data["collective_status"]["gate_below"])

    def test_mpcc_up_acceleration_calculates_above_hover_thrust(self):
        controller = self._controller()
        command = FlightCommand(
            tracking=True,
            gate_vertical_bearing_deg=-10.0,
            gate_vertical_control_active=True,
            vertical_accel_down_mps2=-2.0,
        )

        thrust = controller._calculate_collective(
            command, altitude_correction=0.0, altitude_observable=False
        )
        expected = 0.28 * (1.0 + 2.0 / MPCCGateController.GRAVITY_MPS2)
        self.assertAlmostEqual(thrust, expected)
        self.assertGreater(thrust, 0.28)
        self.assertTrue(controller.data["collective_status"]["gate_above"])

    def test_missing_gate_does_not_reuse_vertical_acceleration(self):
        controller = self._controller()
        command = FlightCommand(
            roll_deg=12.0,
            pitch_deg=-9.0,
            tracking=False,
            gate_vertical_control_active=False,
            vertical_accel_down_mps2=None,
        )

        thrust = controller._calculate_collective(
            command, altitude_correction=0.0, altitude_observable=False
        )
        self.assertAlmostEqual(thrust, 0.28)
        self.assertEqual(
            controller.data["collective_status"]["vertical_source"],
            "gravity_feedforward",
        )

    def test_broad_actuator_minimum_still_bounds_invalid_acceleration(self):
        controller = self._controller()
        command = FlightCommand(
            tracking=True,
            gate_vertical_bearing_deg=20.0,
            gate_vertical_control_active=True,
            vertical_accel_down_mps2=20.0,
        )

        thrust = controller._calculate_collective(
            command, altitude_correction=0.0, altitude_observable=False
        )
        self.assertAlmostEqual(thrust, 0.25)
        self.assertTrue(controller.data["collective_status"]["minimum_applied"])


if __name__ == "__main__":
    unittest.main()
