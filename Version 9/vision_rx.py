import math
import socket
import struct
import threading
import time
from collections import deque

import cv2
import numpy as np

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600

GATE_OUTER_WIDTH_M = 2.7
GATE_OUTER_HEIGHT_M = 2.7
GATE_INNER_WIDTH_M = 1.5
GATE_INNER_HEIGHT_M = 1.5
DRONE_WIDTH_M = 0.28
DRONE_HEIGHT_M = 0.16

GATE_INNER_TO_OUTER_AREA = (
    (GATE_INNER_WIDTH_M * GATE_INNER_HEIGHT_M) /
    (GATE_OUTER_WIDTH_M * GATE_OUTER_HEIGHT_M)
)
GATE_FRAME_TO_OUTER_AREA = 1.0 - GATE_INNER_TO_OUTER_AREA
MIN_GATE_CONTOUR_AREA_RATIO = 0.003
MIN_ORANGE_GATE_AREA_RATIO = 0.002
MAX_ORANGE_GATE_ASPECT = 1.75
GATE_DETECTION_HOLD_S = 0.40
FRONT_FACING_GATE_CORRECTION = True
VISION_DEBUG_INTERVAL_S = 0.5
SHOW_FPV_OVERLAY = True
FPV_OVERLAY_WINDOW = "FPV Gate Overlay"
METRIC_HISTORY_LEN = 120
# size_ratio when the gate roughly fills the frame at this standoff distance.
GATE_DISTANCE_REFERENCE_SIZE = 0.85
GATE_DISTANCE_REFERENCE_M = 2.5
GATE_DISTANCE_GRAPH_MAX_M = 30.0
# Approximate horizontal FOV for the simulator FPV camera (used to build K for PnP).
CAMERA_HFOV_DEG = 90.0
PNP_MAX_REPROJ_ERROR_PX = 18.0
PNP_MIN_DEPTH_M = 0.5
PNP_MAX_DEPTH_M = 50.0

# HSV range for the simulator's orange gate. Keep the hue range broad enough for
# JPEG compression and lighting changes, but require saturation so white/gray
# track elements are ignored.
ORANGE_HSV_LOWER = np.array([3, 80, 70], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)

# AprilTag-inspired pipeline: grayscale -> decimate -> adaptive threshold ->
# segmentation -> quad detection -> decode orientation -> subpixel quad fit.
USE_APRILTAG_STYLE_PIPELINE = True
APRILTAG_DECIMATE = 1
APRILTAG_ADAPTIVE_BLOCK_SIZE = 21
APRILTAG_ADAPTIVE_C = 4
APRILTAG_SUBPIX_WINDOW = (2, 2)
APRILTAG_SUBPIX_ZERO_ZONE = (-1, -1)
APRILTAG_DECODE_WHITE_THRESHOLD = 180
APRILTAG_DECODE_MIN_CONFIDENCE = 0.35
CORNER_LABELS = ("TL", "TR", "BR", "BL")

class VisionRX:

    def __init__(self, data):
        self.data = data
        self.last_vision_debug_print_time = 0.0
        self._overlay_window_ready = False
        self._metric_history = deque(maxlen=METRIC_HISTORY_LEN)
        self._last_gate_detection = None
        self.thread = threading.Thread(
            target=self._vision_loop,
            daemon=True
        )
        self.is_running = True
        self.thread.start()

    def get_thread_for_join(self):
        self.is_running = False
        return self.thread

    def _vision_loop(self):
        header_format = "<IHHIIQ"
        header_sz = struct.calcsize(header_format)
        frames = {}  # frame_id -> received associated frame data

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT))
        print("Listening for camera frames...")

        while self.is_running:
            packet, addr = sock.recvfrom(65536)  # max UDP size

            header = packet[:header_sz]
            payload = packet[header_sz:]

            # frame_id - identifier for this vision frame
            # chunk_id - identifier for this chunk packet of data of this frame
            # total_chunks - total number of chunk packets that make up this frame
            # jpeg_size - full size of jpeg data
            # payload_size - size of this packet
            # sim_time_ns - frame's epoch timestamp in ns on the server
            frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack(header_format, header)

            if frame_id not in frames:
                frames[frame_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "size": jpeg_size,
                    "time": sim_time_ns
                }

            frames[frame_id]["chunks"][chunk_id] = payload

            # Check if frame is complete
            if len(frames[frame_id]["chunks"]) == total_chunks:
                jpeg_bytes = bytearray()

                frame_complete = True
                for i in range(total_chunks):
                    if i not in frames[frame_id]["chunks"]:
                        print('Missing packet %s in frame %s' % (i, frame_id,))
                        frame_complete = False
                        continue
                    jpeg_bytes.extend(frames[frame_id]["chunks"][i])

                if not frame_complete:
                    del frames[frame_id]
                    continue

                img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    self.process_frame(frame_id, image)
                else:
                    print(f"Failed to decode frame: {frame_id}")

                del frames[frame_id]

    def process_frame(self, frame_id, img):
        #
        #
        # Success!
        # image is your FPV camera frame in JPEG format
        #
        #
        detection = self.detect_gate(img)
        detection["frame_id"] = frame_id
        detection["time"] = time.time()
        if detection.get("detected"):
            size_ratio = float(detection.get("size_ratio", 0.0))
            detection["distance_estimate_m"] = self.estimate_gate_distance_m(size_ratio)
        else:
            detection["distance_estimate_m"] = None
        detection = self.stabilize_gate_detection(detection)
        self._record_metric_sample(detection)
        self.data["gate_detection"] = detection
        self.print_gate_debug(detection)
        if SHOW_FPV_OVERLAY:
            overlay = self.draw_gate_overlay(img, detection)
            self.data["fpv_overlay"] = overlay
            self.show_fpv_overlay(overlay)

    def show_fpv_overlay(self, overlay):
        if not self._overlay_window_ready:
            cv2.namedWindow(FPV_OVERLAY_WINDOW, cv2.WINDOW_NORMAL)
            self._overlay_window_ready = True
        cv2.imshow(FPV_OVERLAY_WINDOW, overlay)
        cv2.waitKey(1)

    def draw_gate_overlay(self, img, detection):
        overlay = img.copy()
        height, width = overlay.shape[:2]
        center = (width // 2, height // 2)

        cv2.drawMarker(
            overlay,
            center,
            (220, 220, 220),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=1,
        )

        if not detection.get("detected"):
            self._draw_overlay_hud(overlay, detection, aligned=False)
        else:
            corners = detection.get("corners") or []
            edges = detection.get("edges") or []
            corner_colors = {
                "TL": (0, 255, 255),
                "TR": (0, 200, 255),
                "BR": (0, 140, 255),
                "BL": (0, 80, 255),
            }
            edge_colors = {
                "top": (0, 255, 0),
                "right": (255, 180, 0),
                "bottom": (0, 180, 255),
                "left": (255, 0, 180),
            }

            for edge in edges:
                p1 = tuple(int(round(v)) for v in edge["p1"])
                p2 = tuple(int(round(v)) for v in edge["p2"])
                color = edge_colors.get(edge.get("label"), (0, 255, 0))
                cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

            for label, point in corners:
                pt = (int(round(point[0])), int(round(point[1])))
                color = corner_colors.get(label, (0, 255, 255))
                cv2.circle(overlay, pt, 5, color, -1, lineType=cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    label,
                    (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            gate_center = (
                int(round(detection.get("center_x", center[0]))),
                int(round(detection.get("center_y", center[1]))),
            )
            cv2.circle(overlay, gate_center, 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.line(overlay, center, gate_center, (255, 255, 255), 1, cv2.LINE_AA)

            orientation_deg = float(detection.get("orientation_deg", 0.0))
            top_edge = next((edge for edge in edges if edge.get("label") == "top"), None)
            if top_edge is not None:
                mid = (
                    int(round((top_edge["p1"][0] + top_edge["p2"][0]) * 0.5)),
                    int(round((top_edge["p1"][1] + top_edge["p2"][1]) * 0.5)),
                )
                arrow_len = 42
                angle_rad = math.radians(orientation_deg)
                tip = (
                    int(round(mid[0] + arrow_len * math.cos(angle_rad))),
                    int(round(mid[1] + arrow_len * math.sin(angle_rad))),
                )
                cv2.arrowedLine(
                    overlay,
                    mid,
                    tip,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.35,
                )

            offset_x = float(detection.get("offset_x", 0.0))
            offset_y = float(detection.get("offset_y", 0.0))
            if detection.get("pnp_valid"):
                offset_x = float(detection.get("pnp_offset_x", offset_x))
                offset_y = float(detection.get("pnp_offset_y", offset_y))
                orientation_deg = float(detection.get("pnp_orientation_deg", orientation_deg))
            aligned = (
                abs(offset_x) < 0.12
                and abs(offset_y) < 0.12
                and abs(orientation_deg) < 6.0
            )
            self._draw_overlay_hud(overlay, detection, aligned=aligned)

        self._draw_detection_graph_panel(overlay, detection)
        self._draw_pnp_graph_panel(overlay, detection)
        return overlay

    def _draw_overlay_hud(self, overlay, detection, aligned):
        height, width = overlay.shape[:2]
        panel_w = 220
        panel_h = 150
        x0 = width - panel_w - 12
        y0 = height - panel_h - 12
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + panel_w, y0 + panel_h),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + panel_w, y0 + panel_h),
            (80, 80, 80),
            1,
        )

        if detection.get("detected"):
            orientation_deg = float(detection.get("orientation_deg", 0.0))
            offset_x = float(detection.get("offset_x", 0.0))
            offset_y = float(detection.get("offset_y", 0.0))
            facing = self.describe_gate_facing(orientation_deg, offset_x, offset_y)
            status = "ALIGNED" if aligned else "ADJUST"
            status_color = (80, 220, 80) if aligned else (80, 180, 255)
            lines = [
                ("Gate overlay", (220, 220, 220)),
                (f"tilt {orientation_deg:+.1f} deg", (0, 255, 255)),
                (f"offset x={offset_x:+.2f} y={offset_y:+.2f}", (200, 200, 200)),
                (f"facing: {facing}", (180, 220, 255)),
                (status, status_color),
            ]
        else:
            lines = [
                ("Gate overlay", (220, 220, 220)),
                ("no gate", (120, 120, 120)),
            ]

        for index, (text, color) in enumerate(lines):
            cv2.putText(
                overlay,
                text,
                (x0 + 10, y0 + 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        if detection.get("detected"):
            self._draw_mini_gate_diagram(
                overlay,
                (x0 + panel_w - 78, y0 + 24),
                float(detection.get("orientation_deg", 0.0)),
                float(detection.get("offset_x", 0.0)),
                float(detection.get("offset_y", 0.0)),
            )

    @staticmethod
    def describe_gate_facing(orientation_deg, offset_x, offset_y):
        roll_hint = "level"
        if orientation_deg > 4.0:
            roll_hint = "roll right"
        elif orientation_deg < -4.0:
            roll_hint = "roll left"

        lateral = "centered"
        if offset_x > 0.12:
            lateral = "gate right"
        elif offset_x < -0.12:
            lateral = "gate left"

        vertical = "on line"
        if offset_y > 0.12:
            vertical = "too high"
        elif offset_y < -0.12:
            vertical = "too low"

        return f"{lateral}, {vertical}, {roll_hint}"

    @staticmethod
    def _draw_mini_gate_diagram(overlay, origin, orientation_deg, offset_x, offset_y):
        ox, oy = origin
        size = 28
        angle_rad = math.radians(orientation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        half = size * 0.5
        local = [
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half),
        ]
        points = []
        for lx, ly in local:
            x = ox + lx * cos_a - ly * sin_a
            y = oy + lx * sin_a + ly * cos_a
            points.append((int(round(x)), int(round(y))))
        for index in range(4):
            cv2.line(
                overlay,
                points[index],
                points[(index + 1) % 4],
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        drone_x = int(round(ox - offset_x * 18))
        drone_y = int(round(oy + offset_y * 18))
        cv2.circle(overlay, (drone_x, drone_y), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (ox, oy), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (drone_x, drone_y), (ox, oy), (180, 180, 180), 1, cv2.LINE_AA)

    @staticmethod
    def estimate_gate_distance_m(
            size_ratio,
            reference_size=GATE_DISTANCE_REFERENCE_SIZE,
            reference_distance_m=GATE_DISTANCE_REFERENCE_M):
        if size_ratio is None or size_ratio <= 0.01:
            return None
        return float(reference_distance_m * (reference_size / size_ratio))

    def _record_metric_sample(self, detection):
        if detection.get("detected"):
            pnp_valid = detection.get("pnp_valid")
            self._metric_history.append({
                "distance_m": detection.get("distance_estimate_m"),
                "angle_deg": float(detection.get("orientation_deg", 0.0)),
                "offset_x": float(detection.get("offset_x", 0.0)),
                "offset_y": float(detection.get("offset_y", 0.0)),
                "pnp_distance_m": detection.get("pnp_distance_m") if pnp_valid else None,
                "pnp_angle_deg": float(detection.get("pnp_orientation_deg", 0.0)) if pnp_valid else None,
                "pnp_offset_x": float(detection.get("pnp_offset_x", 0.0)) if pnp_valid else None,
                "pnp_offset_y": float(detection.get("pnp_offset_y", 0.0)) if pnp_valid else None,
            })
        else:
            self._metric_history.append({
                "distance_m": None,
                "angle_deg": 0.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "pnp_distance_m": None,
                "pnp_angle_deg": None,
                "pnp_offset_x": None,
                "pnp_offset_y": None,
            })

    def _draw_detection_graph_panel(self, overlay, detection):
        height, width = overlay.shape[:2]
        panel_w = 200
        panel_h = 210
        x0 = 12
        y0 = 12
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)
        cv2.putText(
            overlay,
            "Detection metrics",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        if detection.get("detected"):
            distance_m = detection.get("distance_estimate_m")
            angle_deg = float(detection.get("orientation_deg", 0.0))
            offset_x = float(detection.get("offset_x", 0.0))
            offset_y = float(detection.get("offset_y", 0.0))
            distance_text = f"{distance_m:.1f} m" if distance_m is not None else "n/a"
        else:
            distance_m = None
            angle_deg = 0.0
            offset_x = 0.0
            offset_y = 0.0
            distance_text = "no gate"

        rows = [
            ("distance", distance_text, distance_m, 0.0, GATE_DISTANCE_GRAPH_MAX_M, (100, 220, 255)),
            ("angle", f"{angle_deg:+.1f} deg", angle_deg, -45.0, 45.0, (0, 255, 255)),
            ("offset x", f"{offset_x:+.2f}", offset_x, -1.0, 1.0, (180, 255, 180)),
            ("offset y", f"{offset_y:+.2f}", offset_y, -1.0, 1.0, (255, 180, 180)),
        ]

        bar_x = x0 + 78
        bar_w = panel_w - 90
        row_y = y0 + 36
        for label, value_text, raw_value, min_value, max_value, color in rows:
            cv2.putText(
                overlay,
                label,
                (x0 + 8, row_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            if raw_value is not None:
                self._draw_metric_bar(
                    overlay,
                    bar_x,
                    row_y,
                    bar_w,
                    12,
                    raw_value,
                    min_value,
                    max_value,
                    color,
                )
            else:
                cv2.rectangle(
                    overlay,
                    (bar_x, row_y),
                    (bar_x + bar_w, row_y + 12),
                    (45, 45, 45),
                    -1,
                )
            cv2.putText(
                overlay,
                value_text,
                (x0 + 8, row_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )
            row_y += 38

        spark_y = y0 + panel_h - 48
        cv2.putText(
            overlay,
            "history",
            (x0 + 8, spark_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )
        spark_h = 10
        spark_w = (panel_w - 24) // 4
        history_keys = ("distance_m", "angle_deg", "offset_x", "offset_y")
        spark_colors = (
            (100, 220, 255),
            (0, 255, 255),
            (180, 255, 180),
            (255, 180, 180),
        )
        spark_ranges = (
            (0.0, GATE_DISTANCE_GRAPH_MAX_M),
            (-45.0, 45.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
        )
        for index, (key, color, value_range) in enumerate(
                zip(history_keys, spark_colors, spark_ranges)):
            sx = x0 + 8 + index * (spark_w + 4)
            self._draw_metric_sparkline(
                overlay,
                sx,
                spark_y,
                spark_w,
                spark_h,
                [sample.get(key) for sample in self._metric_history],
                value_range[0],
                value_range[1],
                color,
            )

    def _draw_pnp_graph_panel(self, overlay, detection):
        height, width = overlay.shape[:2]
        panel_w = 200
        panel_h = 210
        x0 = 12
        y0 = 12 + 210 + 8
        if y0 + panel_h > height - 12:
            y0 = max(12, height - panel_h - 12)

        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

        pnp_valid = detection.get("pnp_valid")
        status_color = (80, 220, 120) if pnp_valid else (120, 120, 120)
        status_text = "valid" if pnp_valid else "invalid"
        cv2.putText(
            overlay,
            f"PnP pose ({status_text})",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            status_color,
            1,
            cv2.LINE_AA,
        )

        if pnp_valid:
            distance_m = detection.get("pnp_distance_m")
            angle_deg = float(detection.get("pnp_orientation_deg", 0.0))
            offset_x = float(detection.get("pnp_offset_x", 0.0))
            offset_y = float(detection.get("pnp_offset_y", 0.0))
            reproj = detection.get("pnp_reproj_error_px")
            distance_text = f"{distance_m:.1f} m" if distance_m is not None else "n/a"
            reproj_text = f"reproj {reproj:.1f}px" if reproj is not None else ""
        else:
            distance_m = None
            angle_deg = 0.0
            offset_x = 0.0
            offset_y = 0.0
            distance_text = "n/a"
            reproj_text = ""

        rows = [
            ("pnp dist", distance_text, distance_m, 0.0, GATE_DISTANCE_GRAPH_MAX_M, (255, 200, 100)),
            ("pnp angle", f"{angle_deg:+.1f} deg", angle_deg if pnp_valid else None, -45.0, 45.0, (255, 220, 80)),
            ("pnp off x", f"{offset_x:+.2f}", offset_x if pnp_valid else None, -1.0, 1.0, (255, 180, 140)),
            ("pnp off y", f"{offset_y:+.2f}", offset_y if pnp_valid else None, -1.0, 1.0, (255, 140, 140)),
        ]

        bar_x = x0 + 78
        bar_w = panel_w - 90
        row_y = y0 + 36
        for label, value_text, raw_value, min_value, max_value, color in rows:
            cv2.putText(
                overlay,
                label,
                (x0 + 8, row_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            if raw_value is not None:
                self._draw_metric_bar(
                    overlay,
                    bar_x,
                    row_y,
                    bar_w,
                    12,
                    raw_value,
                    min_value,
                    max_value,
                    color,
                )
            else:
                cv2.rectangle(
                    overlay,
                    (bar_x, row_y),
                    (bar_x + bar_w, row_y + 12),
                    (45, 45, 45),
                    -1,
                )
            cv2.putText(
                overlay,
                value_text,
                (x0 + 8, row_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color if raw_value is not None else (100, 100, 100),
                1,
                cv2.LINE_AA,
            )
            row_y += 38

        if reproj_text:
            cv2.putText(
                overlay,
                reproj_text,
                (x0 + 8, row_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )

        spark_y = y0 + panel_h - 48
        cv2.putText(
            overlay,
            "pnp history",
            (x0 + 8, spark_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )
        spark_h = 10
        spark_w = (panel_w - 24) // 4
        history_keys = ("pnp_distance_m", "pnp_angle_deg", "pnp_offset_x", "pnp_offset_y")
        spark_colors = (
            (255, 200, 100),
            (255, 220, 80),
            (255, 180, 140),
            (255, 140, 140),
        )
        spark_ranges = (
            (0.0, GATE_DISTANCE_GRAPH_MAX_M),
            (-45.0, 45.0),
            (-1.0, 1.0),
            (-1.0, 1.0),
        )
        for index, (key, color, value_range) in enumerate(
                zip(history_keys, spark_colors, spark_ranges)):
            sx = x0 + 8 + index * (spark_w + 4)
            self._draw_metric_sparkline(
                overlay,
                sx,
                spark_y,
                spark_w,
                spark_h,
                [sample.get(key) for sample in self._metric_history],
                value_range[0],
                value_range[1],
                color,
            )

    @staticmethod
    def _draw_metric_bar(overlay, x, y, width, height, value, min_value, max_value, color):
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (45, 45, 45), -1)
        mid_x = x + width // 2
        cv2.line(overlay, (mid_x, y), (mid_x, y + height), (70, 70, 70), 1)
        span = max(max_value - min_value, 1e-6)
        clamped = max(min_value, min(max_value, float(value)))
        if min_value < 0.0 < max_value:
            zero_x = x + int(round((0.0 - min_value) / span * width))
            value_x = x + int(round((clamped - min_value) / span * width))
            bar_left = min(zero_x, value_x)
            bar_right = max(zero_x, value_x)
        else:
            bar_left = x
            bar_right = x + int(round((clamped - min_value) / span * width))
        if bar_right > bar_left:
            cv2.rectangle(
                overlay,
                (bar_left, y + 2),
                (bar_right, y + height - 2),
                color,
                -1,
            )

    @staticmethod
    def _draw_metric_sparkline(overlay, x, y, width, height, values, min_value, max_value, color):
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (35, 35, 35), -1)
        valid = [float(v) for v in values if v is not None]
        if len(valid) < 2:
            return
        span = max(max_value - min_value, 1e-6)
        step = width / max(len(valid) - 1, 1)
        points = []
        start_index = max(0, len(valid) - int(width))
        clipped = valid[start_index:]
        for index, value in enumerate(clipped):
            px = x + int(round(index * step))
            norm = (max(min_value, min(max_value, value)) - min_value) / span
            py = y + height - 1 - int(round(norm * (height - 2)))
            points.append((px, py))
        for index in range(1, len(points)):
            cv2.line(overlay, points[index - 1], points[index], color, 1, cv2.LINE_AA)

    def stabilize_gate_detection(self, detection):
        now = float(detection.get("time", time.time()))
        if detection.get("detected"):
            detection["raw_detected"] = True
            detection["held_detection"] = False
            self._last_gate_detection = detection.copy()
            return detection

        if self._last_gate_detection is None:
            detection["raw_detected"] = False
            detection["held_detection"] = False
            return detection

        last_time = float(self._last_gate_detection.get("time", 0.0))
        hold_age = now - last_time
        if hold_age < 0.0 or hold_age > GATE_DETECTION_HOLD_S:
            self._last_gate_detection = None
            detection["raw_detected"] = False
            detection["held_detection"] = False
            return detection

        held = self._last_gate_detection.copy()
        held["frame_id"] = detection.get("frame_id")
        held["time"] = now
        held["raw_detected"] = False
        held["held_detection"] = True
        held["orange_candidates"] = detection.get("orange_candidates", held.get("orange_candidates", 0))
        held["square_candidates"] = detection.get("square_candidates", held.get("square_candidates", 0))
        held["best_rejected_score"] = detection.get("best_rejected_score", held.get("best_rejected_score", 0.0))
        held["confidence"] = max(0.45, float(held.get("confidence", 0.0)) * 0.9)
        return held

    def detect_gate(self, img):
        height, width = img.shape[:2]
        orange_candidates = self.find_orange_gate_candidates(img)
        square_candidates = self.find_square_candidates(img)
        apriltag_candidates = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        decode_confidence = 0.0
        pipeline_stage = "legacy"

        if USE_APRILTAG_STYLE_PIPELINE:
            gray, binary, binary_inv, scale = self.apriltag_preprocess(img)
            apriltag_candidates = self.find_quad_candidates_apriltag(
                binary,
                binary_inv,
                width,
                height,
                scale,
            )
            gate, best_rejected_score = self.choose_decoded_gate_candidate(
                apriltag_candidates,
                orange_candidates,
                gray,
                width,
                height,
            )
            pipeline_stage = "apriltag"
        else:
            gate, best_rejected_score = self.choose_orange_gate_candidate(
                orange_candidates,
                width,
                height,
            )

        if gate is None:
            gate, best_rejected_score = self.choose_orange_gate_candidate(
                orange_candidates,
                width,
                height,
            )
            pipeline_stage = "orange_fallback"

        if gate is None:
            gate, best_rejected_score = self.choose_square_gate_candidate(
                square_candidates,
                width,
                height,
            )
            pipeline_stage = "edge_fallback"

        if gate is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "orange_candidates": len(orange_candidates),
                "square_candidates": len(square_candidates),
                "apriltag_candidates": len(apriltag_candidates),
                "best_rejected_score": best_rejected_score,
                "pipeline_stage": pipeline_stage,
            }

        geometry_corners, corner_source = self.refine_gate_corners(gate, square_candidates)
        if USE_APRILTAG_STYLE_PIPELINE:
            geometry_corners = self.fit_external_quad(gray, geometry_corners)
            corner_source = f"{corner_source}+subpix"

        labeled_corners, decode_confidence = self.decode_gate_corners(
            geometry_corners,
            gate.get("inner_corners"),
            gray,
        )
        decode_confidence = max(
            decode_confidence,
            float(gate.get("decode_confidence", 0.0)),
        )
        perspective_asymmetry = self.corner_perspective_asymmetry(
            np.array([pt for _, pt in labeled_corners], dtype=np.float32),
        )
        use_front_square = (
            FRONT_FACING_GATE_CORRECTION
            and decode_confidence < APRILTAG_DECODE_MIN_CONFIDENCE
        )
        if use_front_square:
            raw_corners = np.array(
                [pt for _, pt in labeled_corners],
                dtype=np.float32,
            )
            geometry_corners = self.front_facing_square_corners(raw_corners)
            labeled_corners = self.decode_gate_corners(
                geometry_corners,
                gate.get("inner_corners"),
                gray,
            )[0]
            corner_source = f"{corner_source}+front_square"

        geometry = self.gate_geometry_from_corners(
            np.array([pt for _, pt in labeled_corners], dtype=np.float32),
        )
        geometry["corners"] = labeled_corners
        resolve_ambiguity = decode_confidence < APRILTAG_DECODE_MIN_CONFIDENCE
        pnp_pose = self.estimate_gate_pose_pnp(
            labeled_corners,
            width,
            height,
            resolve_ambiguity=resolve_ambiguity,
        )
        center_x, center_y = gate["center"]
        size_px = gate["size_px"]
        size_ratio = size_px / max(1.0, min(width, height))
        distance_estimate_m = self.estimate_gate_distance_m(size_ratio)
        if pnp_pose.get("pnp_valid"):
            distance_estimate_m = pnp_pose["pnp_distance_m"]
        return {
            "detected": True,
            "confidence": gate["confidence"],
            "match_type": gate["match_type"],
            "corner_source": corner_source,
            "pipeline_stage": pipeline_stage,
            "decode_confidence": decode_confidence,
            "perspective_asymmetry": perspective_asymmetry,
            "orange_candidates": len(orange_candidates),
            "square_candidates": len(square_candidates),
            "apriltag_candidates": len(apriltag_candidates),
            "best_rejected_score": best_rejected_score,
            "center_x": center_x,
            "center_y": center_y,
            "offset_x": (center_x - (width * 0.5)) / (width * 0.5),
            "offset_y": (center_y - (height * 0.5)) / (height * 0.5),
            "size_px": size_px,
            "size_ratio": size_ratio,
            "distance_estimate_m": distance_estimate_m,
            "image_width": width,
            "image_height": height,
            "inner_clearance_m": min(
                GATE_INNER_WIDTH_M - DRONE_WIDTH_M,
                GATE_INNER_HEIGHT_M - DRONE_HEIGHT_M,
            ),
            "corners": geometry["corners"],
            "edges": geometry["edges"],
            "vertex_angles": geometry["vertex_angles"],
            "orientation_deg": geometry["orientation_deg"],
            **pnp_pose,
        }

    @staticmethod
    def extract_quad_corners(contour):
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        if perimeter <= 0.0:
            return None

        for epsilon_factor in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, epsilon_factor * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)

        return None

    @staticmethod
    def apriltag_preprocess(img, decimate=APRILTAG_DECIMATE):
        """Grayscale -> decimate -> adaptive threshold (AprilTag stages 1-3)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if decimate > 1:
            small_w = max(1, width // decimate)
            small_h = max(1, height // decimate)
            small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small = gray

        blurred = cv2.GaussianBlur(small, (3, 3), 0)
        block = APRILTAG_ADAPTIVE_BLOCK_SIZE | 1
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            APRILTAG_ADAPTIVE_C,
        )
        binary_inv = cv2.bitwise_not(binary)
        return gray, binary, binary_inv, float(decimate)

    def find_quad_candidates_apriltag(self, binary, binary_inv, full_w, full_h, scale):
        """Segmentation + quad detection on adaptive-threshold images."""
        candidates = []
        for mask in (binary, binary_inv):
            candidates.extend(
                self._find_quads_in_binary(mask, full_w, full_h, scale)
            )
        return self._dedupe_quad_candidates(candidates)

    def _find_quads_in_binary(self, binary, full_w, full_h, scale):
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if hierarchy is None:
            return []

        min_area = full_w * full_h * MIN_GATE_CONTOUR_AREA_RATIO / (scale * scale)
        candidates = []
        hierarchy = hierarchy[0]
        for index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            quad_corners = self.extract_quad_corners(contour)
            if quad_corners is None:
                continue

            quad_corners = (quad_corners * scale).astype(np.float32)
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            if rect_width <= 1.0 or rect_height <= 1.0:
                continue

            rect_width *= scale
            rect_height *= scale
            aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
            if aspect > 1.35:
                continue

            rectangularity = (area * scale * scale) / (rect_width * rect_height)
            if rectangularity < 0.55:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0.0:
                continue

            center = (
                moments["m10"] / moments["m00"] * scale,
                moments["m01"] / moments["m00"] * scale,
            )
            candidates.append({
                "contour": quad_corners.astype(np.float32),
                "corners": quad_corners,
                "area": area * scale * scale,
                "center": center,
                "size_px": max(rect_width, rect_height),
                "aspect": aspect,
                "rectangularity": rectangularity,
                "parent_index": hierarchy[index][3],
            })
        return candidates

    @staticmethod
    def _dedupe_quad_candidates(candidates):
        if not candidates:
            return []

        candidates = sorted(candidates, key=lambda item: item["area"], reverse=True)
        kept = []
        for candidate in candidates:
            duplicate = False
            for existing in kept:
                center_error = VisionRX.normalized_center_error(
                    candidate["center"],
                    existing["center"],
                    max(candidate["size_px"], existing["size_px"]),
                )
                if center_error < 0.12:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        return kept

    def choose_decoded_gate_candidate(
        self,
        apriltag_candidates,
        orange_candidates,
        gray,
        image_width,
        image_height,
    ):
        """Inner/outer quad pairing + orange validation (decode-ID equivalent)."""
        best_gate = None
        best_score = 0.0

        for outer in apriltag_candidates:
            for inner in apriltag_candidates:
                if inner is outer or inner["area"] >= outer["area"]:
                    continue
                if cv2.pointPolygonTest(
                    outer["contour"],
                    inner["center"],
                    False,
                ) < 0:
                    continue

                area_ratio = inner["area"] / outer["area"]
                ratio_error = abs(area_ratio - GATE_INNER_TO_OUTER_AREA)
                if ratio_error > 0.22:
                    continue

                center_error = self.normalized_center_error(
                    inner["center"],
                    outer["center"],
                    outer["size_px"],
                )
                if center_error > 0.35:
                    continue

                labeled, decode_confidence = self.decode_gate_corners(
                    outer["corners"],
                    inner["corners"],
                    gray,
                )
                orange_score = self._orange_overlap_score(
                    outer["corners"],
                    orange_candidates,
                )
                score = (
                    1.0
                    - min(1.0, ratio_error / 0.22) * 0.30
                    - min(1.0, center_error / 0.35) * 0.20
                    + min(0.20, outer["area"] / (image_width * image_height))
                    + decode_confidence * 0.25
                    + orange_score * 0.15
                )
                if score > best_score:
                    best_score = score
                    best_gate = {
                        "center": outer["center"],
                        "corners": outer["corners"],
                        "inner_corners": inner["corners"],
                        "corner_source": "apriltag_inner_outer",
                        "size_px": outer["size_px"],
                        "confidence": max(0.0, min(1.0, score)),
                        "match_type": "apriltag_decode",
                        "decode_confidence": decode_confidence,
                    }

        if best_gate is not None and best_gate["confidence"] >= 0.45:
            return best_gate, best_score

        for candidate in apriltag_candidates:
            orange_score = self._orange_overlap_score(
                candidate["corners"],
                orange_candidates,
            )
            if orange_score < 0.15:
                continue

            labeled, decode_confidence = self.decode_gate_corners(
                candidate["corners"],
                None,
                gray,
            )
            center_error = self.normalized_center_error(
                candidate["center"],
                (image_width * 0.5, image_height * 0.5),
                min(image_width, image_height),
            )
            score = (
                0.65
                + orange_score * 0.20
                + decode_confidence * 0.15
                - min(1.0, center_error / 0.75) * 0.10
            )
            if score > best_score:
                best_score = score
                best_gate = {
                    "center": candidate["center"],
                    "corners": candidate["corners"],
                    "inner_corners": None,
                    "corner_source": "apriltag_outer",
                    "size_px": candidate["size_px"],
                    "confidence": max(0.0, min(1.0, score)),
                    "match_type": "apriltag_orange",
                    "decode_confidence": decode_confidence,
                }

        if best_gate is not None and best_gate["confidence"] >= 0.45:
            return best_gate, best_score

        return None, best_score

    @staticmethod
    def _orange_overlap_score(outer_corners, orange_candidates):
        if not orange_candidates:
            return 0.0

        outer_poly = outer_corners.astype(np.float32).reshape(-1, 1, 2)
        best = 0.0
        for candidate in orange_candidates:
            center_error = VisionRX.normalized_center_error(
                candidate["center"],
                tuple(np.mean(outer_corners, axis=0)),
                candidate["size_px"],
            )
            if center_error > 0.35:
                continue
            if cv2.pointPolygonTest(outer_poly, candidate["center"], False) < 0:
                continue
            best = max(best, 1.0 - min(1.0, center_error / 0.35))
        return best

    @classmethod
    def decode_gate_corners(cls, outer_corners, inner_corners, gray):
        """
        Resolve TL/TR/BR/BL like AprilTag decodeID:
        inner-outer rotation + asymmetric branding (VQ-01 top-right).
        """
        ordered = cls.order_corners(outer_corners)
        decode_confidence = 0.35

        if inner_corners is not None:
            inner_ordered = cls.order_corners(inner_corners)
            best_rot = cls._best_inner_outer_rotation(ordered, inner_ordered)
            ordered = np.roll(ordered, best_rot, axis=0)
            containment = cls._inner_containment_score(ordered, inner_ordered)
            decode_confidence = max(decode_confidence, 0.45 + containment * 0.15)

        labeled = cls.corner_display_labels(ordered)
        branding_a = cls._gate_branding_orientation_score(gray, labeled)
        labeled_flip = cls.corner_display_labels(np.roll(ordered, 2, axis=0))
        branding_b = cls._gate_branding_orientation_score(gray, labeled_flip)
        if branding_b > branding_a + 0.02:
            ordered = np.roll(ordered, 2, axis=0)
            labeled = labeled_flip
            decode_confidence = max(decode_confidence, 0.55)
        elif branding_a > branding_b + 0.02:
            decode_confidence = max(decode_confidence, 0.55)

        return labeled, min(1.0, decode_confidence)

    @staticmethod
    def _inner_containment_score(outer, inner):
        inside = 0
        for point in inner:
            if cv2.pointPolygonTest(
                outer.astype(np.float32),
                (float(point[0]), float(point[1])),
                False,
            ) >= 0:
                inside += 1
        return inside / max(len(inner), 1)

    @staticmethod
    def _best_inner_outer_rotation(outer, inner):
        best_rot = 0
        best_score = -1.0
        for rot in range(4):
            rotated = np.roll(outer, rot, axis=0)
            score = VisionRX._inner_containment_score(rotated, inner)
            if score > best_score:
                best_score = score
                best_rot = rot
        return best_rot

    @classmethod
    def _gate_branding_orientation_score(cls, gray, labeled_corners):
        """Score orientation using white branding (top / top-right on the gate)."""
        corner_map = {
            label: np.array(point, dtype=np.float32)
            for label, point in labeled_corners
        }
        src = np.array([corner_map[label] for label in CORNER_LABELS], dtype=np.float32)
        size = 120
        dst = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(gray, homography, (size, size))
        white = (warped >= APRILTAG_DECODE_WHITE_THRESHOLD).astype(np.float32)
        top_band = white[: size // 3, :].mean()
        bottom_band = white[2 * size // 3 :, :].mean()
        top_right = white[: size // 2, size // 2 :].mean()
        bottom_left = white[size // 2 :, : size // 2].mean()
        return float((top_right - bottom_left) + 0.35 * (top_band - bottom_band))

    @staticmethod
    def fit_external_quad(gray, corners):
        """Subpixel corner refinement (AprilTag fit external quad stage)."""
        pts = np.array(corners, dtype=np.float32).reshape(4, 1, 2)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.01,
        )
        cv2.cornerSubPix(
            gray,
            pts,
            APRILTAG_SUBPIX_WINDOW,
            APRILTAG_SUBPIX_ZERO_ZONE,
            criteria,
        )
        return pts.reshape(4, 2)

    def refine_gate_corners(self, gate, square_candidates):
        center = gate["center"]
        gate_size = max(float(gate.get("size_px", 1.0)), 1.0)

        best_corners = None
        best_score = 0.0
        for candidate in square_candidates:
            center_error = self.normalized_center_error(
                candidate["center"],
                center,
                candidate["size_px"],
            )
            if center_error > 0.45:
                continue

            size_error = abs(candidate["size_px"] - gate_size) / gate_size
            if size_error > 0.65:
                continue

            score = (1.0 - min(1.0, center_error / 0.45)) * 0.65
            score += (1.0 - min(1.0, size_error / 0.65)) * 0.35
            if score > best_score:
                best_score = score
                best_corners = candidate["corners"]

        if best_corners is not None and best_score >= 0.45:
            return best_corners, "edges"

        if gate.get("corner_source") == "contour":
            return gate["corners"], "contour"

        return gate["corners"], "rect"

    def find_orange_gate_candidates(self, img):
        height, width = img.shape[:2]
        image_area = width * height
        min_orange_pixels = image_area * MIN_ORANGE_GATE_AREA_RATIO

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ORANGE_HSV_LOWER, ORANGE_HSV_UPPER)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            if rect_width <= 2.0 or rect_height <= 2.0:
                continue

            rect_area = rect_width * rect_height
            if rect_area <= 0.0:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            orange_pixels = cv2.countNonZero(mask[y:y + h, x:x + w])
            if orange_pixels < min_orange_pixels:
                continue

            aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
            if aspect > MAX_ORANGE_GATE_ASPECT:
                continue

            fill_ratio = orange_pixels / rect_area
            # The physical gate is an orange square frame: orange area is roughly
            # outer square minus inner square, about 69% of the outer footprint.
            if fill_ratio < 0.18 or fill_ratio > 0.92:
                continue

            center = rect[0]
            quad_corners = self.extract_quad_corners(contour)
            if quad_corners is not None:
                corners = quad_corners
                corner_source = "contour"
            else:
                corners = cv2.boxPoints(rect)
                corner_source = "rect"
            candidates.append({
                "center": center,
                "corners": corners,
                "corner_source": corner_source,
                "size_px": max(rect_width, rect_height),
                "rect_area": rect_area,
                "orange_pixels": orange_pixels,
                "aspect": aspect,
                "fill_ratio": fill_ratio,
            })

        return candidates

    def find_square_candidates(self, img):
        height, width = img.shape[:2]
        min_area = width * height * MIN_GATE_CONTOUR_AREA_RATIO

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0.0:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            rect = cv2.minAreaRect(approx)
            rect_width, rect_height = rect[1]
            if rect_width <= 1.0 or rect_height <= 1.0:
                continue

            aspect = max(rect_width, rect_height) / min(rect_width, rect_height)
            if aspect > 1.35:
                continue

            rectangularity = area / (rect_width * rect_height)
            if rectangularity < 0.55:
                continue

            moments = cv2.moments(approx)
            if moments["m00"] == 0.0:
                continue

            center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            candidates.append({
                "contour": approx,
                "corners": approx.reshape(4, 2),
                "area": area,
                "center": center,
                "size_px": max(rect_width, rect_height),
                "aspect": aspect,
                "rectangularity": rectangularity,
            })

        return candidates

    def choose_orange_gate_candidate(self, candidates, image_width, image_height):
        best_gate = None
        best_score = 0.0
        image_area = image_width * image_height
        image_center = (image_width * 0.5, image_height * 0.5)

        for candidate in candidates:
            aspect_error = abs(candidate["aspect"] - 1.0)
            fill_error = abs(candidate["fill_ratio"] - GATE_FRAME_TO_OUTER_AREA)
            center_error = self.normalized_center_error(
                candidate["center"],
                image_center,
                min(image_width, image_height)
            )
            area_score = min(0.25, candidate["rect_area"] / image_area)
            score = (
                1.0
                - min(1.0, aspect_error / 0.75) * 0.30
                - min(1.0, fill_error / 0.35) * 0.30
                - min(1.0, center_error / 0.75) * 0.20
                + area_score
            )

            if score > best_score:
                best_score = score
                best_gate = {
                    "center": candidate["center"],
                    "corners": candidate["corners"],
                    "corner_source": candidate.get("corner_source", "rect"),
                    "size_px": candidate["size_px"],
                    "confidence": max(0.0, min(1.0, score)),
                    "match_type": "orange_mask",
                }

        if best_gate is not None and best_gate["confidence"] >= 0.45:
            return best_gate, best_score

        return None, best_score

    def choose_square_gate_candidate(self, candidates, image_width, image_height):
        best_gate = None
        best_score = 0.0

        for outer in candidates:
            for inner in candidates:
                if inner is outer or inner["area"] >= outer["area"]:
                    continue
                if cv2.pointPolygonTest(outer["contour"], inner["center"], False) < 0:
                    continue

                area_ratio = inner["area"] / outer["area"]
                ratio_error = abs(area_ratio - GATE_INNER_TO_OUTER_AREA)
                if ratio_error > 0.22:
                    continue

                center_error = self.normalized_center_error(inner["center"], outer["center"], outer["size_px"])
                if center_error > 0.35:
                    continue

                score = (
                    1.0
                    - min(1.0, ratio_error / 0.22) * 0.45
                    - min(1.0, center_error / 0.35) * 0.35
                    + min(0.20, outer["area"] / (image_width * image_height))
                )
                if score > best_score:
                    best_score = score
                    best_gate = {
                        "center": inner["center"],
                        "corners": outer["corners"],
                        "corner_source": "edges",
                        "size_px": inner["size_px"],
                        "confidence": max(0.0, min(1.0, score)),
                        "match_type": "inner_outer",
                    }

        if best_gate is not None and best_gate["confidence"] >= 0.45:
            return best_gate, best_score

        return None, best_score

    def print_gate_debug(self, detection):
        now = time.time()
        if now - self.last_vision_debug_print_time < VISION_DEBUG_INTERVAL_S:
            return

        self.last_vision_debug_print_time = now
        if detection.get("detected"):
            corner_text = self.format_corners(detection.get("corners"))
            edge_text = self.format_edges(detection.get("edges"))
            vertex_text = self.format_vertex_angles(detection.get("vertex_angles"))
            pnp_dist = detection.get("pnp_distance_m")
            pnp_reproj = detection.get("pnp_reproj_error_px")
            pnp_dist_text = f"{pnp_dist:.2f}m" if pnp_dist is not None else "n/a"
            pnp_reproj_text = f"{pnp_reproj:.1f}" if pnp_reproj is not None else "n/a"
            print(
                "\n[Vision]\n"
                f"  gate    : detected  conf={detection.get('confidence', 0.0):.2f}  "
                f"match={detection.get('match_type', 'unknown')}  "
                f"corners={detection.get('corner_source', 'unknown')}\n"
                f"  pipeline: {detection.get('pipeline_stage', 'unknown')}  "
                f"decode={detection.get('decode_confidence', 0.0):.2f}  "
                f"pnp_ambiguity={'yes' if detection.get('pnp_ambiguity_resolved') else 'no'}\n"
                f"  offset  : x={detection.get('offset_x', 0.0):+.2f}  "
                f"y={detection.get('offset_y', 0.0):+.2f}  "
                f"size={detection.get('size_ratio', 0.0):.2f}\n"
                f"  orient  : {detection.get('orientation_deg', 0.0):+.1f} deg  "
                f"(horizontal tilt in image)\n"
                f"  pnp     : "
                f"{'valid' if detection.get('pnp_valid') else 'invalid'}  "
                f"dist={pnp_dist_text}  "
                f"reproj={pnp_reproj_text} px\n"
                f"  corners : {corner_text}\n"
                f"  edges   : {edge_text}\n"
                f"  vertices: {vertex_text}\n"
                f"  samples : orange={detection.get('orange_candidates', 0)}  "
                f"squares={detection.get('square_candidates', 0)}",
                flush=True
            )
        else:
            print(
                "\n[Vision]\n"
                "  gate    : none\n"
                f"  samples : orange={detection.get('orange_candidates', 0)}  "
                f"squares={detection.get('square_candidates', 0)}  "
                f"best_rejected_score={detection.get('best_rejected_score', 0.0):.2f}",
                flush=True
            )

    @staticmethod
    def normalized_center_error(center_a, center_b, scale):
        dx = center_a[0] - center_b[0]
        dy = center_a[1] - center_b[1]
        return float(np.hypot(dx, dy) / max(1.0, scale))

    @staticmethod
    def order_corners(corners):
        pts = np.array(corners, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return pts[np.argsort(angles)]

    @staticmethod
    def corner_display_labels(corners):
        # Order points into a consistent cyclic winding (TL -> TR -> BR -> BL),
        # then anchor the sequence to the top-left-most corner. Assigning labels
        # by winding order guarantees every label is used exactly once, even when
        # the gate is tilted or seen at a steep perspective angle (a quadrant-based
        # assignment can otherwise duplicate one label and drop another).
        ordered = VisionRX.order_corners(corners)
        sums = ordered[:, 0] + ordered[:, 1]
        start = int(np.argmin(sums))
        rolled = np.roll(ordered, -start, axis=0)
        return [
            (CORNER_LABELS[index], (float(rolled[index][0]), float(rolled[index][1])))
            for index in range(4)
        ]

    @staticmethod
    def normalize_tilt_deg(angle_deg):
        while angle_deg <= -90.0:
            angle_deg += 180.0
        while angle_deg > 90.0:
            angle_deg -= 180.0
        return angle_deg

    @classmethod
    def front_facing_square_corners(cls, corners):
        pts = np.array(corners, dtype=np.float32).reshape(4, 2)
        geometry = cls.gate_geometry_from_corners(pts)
        angle_rad = math.radians(geometry["orientation_deg"])
        ux = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
        uy = np.array([-math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32)
        center = pts.mean(axis=0)

        rel = pts - center
        x_coords = rel @ ux
        y_coords = rel @ uy
        span_x = float(x_coords.max() - x_coords.min())
        span_y = float(y_coords.max() - y_coords.min())
        half_size = max(1.0, max(span_x, span_y) * 0.5)

        return np.array([
            center - ux * half_size - uy * half_size,
            center + ux * half_size - uy * half_size,
            center + ux * half_size + uy * half_size,
            center - ux * half_size + uy * half_size,
        ], dtype=np.float32)

    @classmethod
    def corner_perspective_asymmetry(cls, corners):
        geometry = cls.gate_geometry_from_corners(corners)
        edge_map = {edge["label"]: edge for edge in geometry["edges"]}
        left_len = float(edge_map["left"]["length_px"])
        right_len = float(edge_map["right"]["length_px"])
        top_len = float(edge_map["top"]["length_px"])
        bottom_len = float(edge_map["bottom"]["length_px"])
        vertical_asymmetry = abs(left_len - right_len) / max(left_len, right_len, 1.0)
        horizontal_asymmetry = abs(top_len - bottom_len) / max(top_len, bottom_len, 1.0)
        return max(vertical_asymmetry, horizontal_asymmetry)

    @classmethod
    def robust_gate_orientation_deg(cls, edges):
        edge_map = {edge["label"]: edge for edge in edges}
        tilts = [
            cls.normalize_tilt_deg(float(edge_map["top"]["angle_deg"])),
            cls.normalize_tilt_deg(float(edge_map["bottom"]["angle_deg"])),
            cls.normalize_tilt_deg(float(edge_map["right"]["angle_deg"]) - 90.0),
            cls.normalize_tilt_deg(float(edge_map["left"]["angle_deg"]) - 90.0),
        ]
        return float(np.median(np.array(tilts, dtype=np.float32)))

    @classmethod
    def gate_outer_object_points(cls):
        half_w = GATE_OUTER_WIDTH_M * 0.5
        half_h = GATE_OUTER_HEIGHT_M * 0.5
        # Gate plane: X right, Y down, Z forward (normal toward the camera).
        return np.array([
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ], dtype=np.float64)

    @staticmethod
    def camera_matrix(image_width, image_height, hfov_deg=CAMERA_HFOV_DEG):
        fx = image_width / (2.0 * math.tan(math.radians(hfov_deg * 0.5)))
        fy = fx
        cx = image_width * 0.5
        cy = image_height * 0.5
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @classmethod
    def _empty_pnp_pose(cls):
        return {
            "pnp_valid": False,
            "pnp_distance_m": None,
            "pnp_lateral_m": None,
            "pnp_vertical_m": None,
            "pnp_offset_x": None,
            "pnp_offset_y": None,
            "pnp_orientation_deg": None,
            "pnp_reproj_error_px": None,
            "pnp_size_ratio": None,
        }

    @classmethod
    def estimate_gate_pose_pnp(
        cls,
        labeled_corners,
        image_width,
        image_height,
        resolve_ambiguity=True,
    ):
        if resolve_ambiguity:
            return cls._resolve_pnp_ambiguity(
                labeled_corners,
                image_width,
                image_height,
            )
        return cls._solve_pnp_once(labeled_corners, image_width, image_height)

    @classmethod
    def _resolve_pnp_ambiguity(cls, labeled_corners, image_width, image_height):
        """Try four corner permutations; pick pose with best reproj + facing check."""
        invalid = cls._empty_pnp_pose()
        if not labeled_corners or len(labeled_corners) != 4:
            return invalid

        corner_map = {label: pt for label, pt in labeled_corners}
        if not all(label in corner_map for label in CORNER_LABELS):
            return invalid

        base_pts = np.array(
            [corner_map[label] for label in CORNER_LABELS],
            dtype=np.float64,
        )
        best_pose = invalid
        best_cost = float("inf")

        for rot in range(4):
            perm_pts = np.roll(base_pts, rot, axis=0)
            perm_labeled = list(zip(CORNER_LABELS, perm_pts))
            pose = cls._solve_pnp_once(perm_labeled, image_width, image_height)
            if not pose.get("pnp_valid"):
                continue

            cost = float(pose["pnp_reproj_error_px"])
            if pose.get("pnp_normal_facing"):
                cost -= 2.0
            if cost < best_cost:
                best_cost = cost
                best_pose = pose.copy()
                best_pose["pnp_corner_rotation"] = rot
                best_pose["pnp_ambiguity_resolved"] = True

        if best_pose.get("pnp_valid"):
            return best_pose
        return cls._solve_pnp_once(labeled_corners, image_width, image_height)

    @classmethod
    def _solve_pnp_once(cls, labeled_corners, image_width, image_height):
        invalid = cls._empty_pnp_pose()
        if not labeled_corners or len(labeled_corners) != 4:
            return invalid

        corner_map = {label: pt for label, pt in labeled_corners}
        if not all(label in corner_map for label in CORNER_LABELS):
            return invalid

        image_points = np.array(
            [corner_map[label] for label in CORNER_LABELS],
            dtype=np.float64,
        )
        object_points = cls.gate_outer_object_points()
        camera_matrix = cls.camera_matrix(image_width, image_height)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        if not ok:
            return invalid

        tvec = tvec.reshape(3)
        depth = float(tvec[2])
        if depth < PNP_MIN_DEPTH_M or depth > PNP_MAX_DEPTH_M:
            return invalid

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        reproj_error = float(
            np.linalg.norm(
                projected.reshape(4, 2) - image_points,
                axis=1,
            ).mean()
        )
        if reproj_error > PNP_MAX_REPROJ_ERROR_PX:
            return invalid

        rotation, _ = cv2.Rodrigues(rvec)
        lateral_m = float(tvec[0])
        vertical_m = float(tvec[1])
        distance_m = depth
        pnp_size_ratio = GATE_DISTANCE_REFERENCE_SIZE * GATE_DISTANCE_REFERENCE_M / distance_m
        normal_facing = float(rotation[2, 2]) < 0.0
        return {
            "pnp_valid": True,
            "pnp_distance_m": distance_m,
            "pnp_lateral_m": lateral_m,
            "pnp_vertical_m": vertical_m,
            "pnp_offset_x": lateral_m / distance_m,
            "pnp_offset_y": vertical_m / distance_m,
            "pnp_orientation_deg": cls.normalize_tilt_deg(
                math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
            ),
            "pnp_reproj_error_px": reproj_error,
            "pnp_size_ratio": pnp_size_ratio,
            "pnp_normal_facing": normal_facing,
            "pnp_ambiguity_resolved": False,
        }

    @classmethod
    def gate_geometry_from_corners(cls, corners):
        ordered = cls.order_corners(corners)
        raw_edges = []
        for index in range(4):
            p1 = ordered[index]
            p2 = ordered[(index + 1) % 4]
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            raw_edges.append({
                "p1": (float(p1[0]), float(p1[1])),
                "p2": (float(p2[0]), float(p2[1])),
                "mid_x": float((p1[0] + p2[0]) * 0.5),
                "mid_y": float((p1[1] + p2[1]) * 0.5),
                "length_px": float(np.hypot(dx, dy)),
                "angle_deg": float(math.degrees(math.atan2(dy, dx))),
            })

        edges_by_y = sorted(raw_edges, key=lambda edge: edge["mid_y"])
        top = edges_by_y[0]
        bottom = edges_by_y[-1]
        side_edges = edges_by_y[1:3]
        left = min(side_edges, key=lambda edge: edge["mid_x"])
        right = max(side_edges, key=lambda edge: edge["mid_x"])
        edges = [
            {**top, "label": "top"},
            {**right, "label": "right"},
            {**bottom, "label": "bottom"},
            {**left, "label": "left"},
        ]

        orientation_deg = cls.robust_gate_orientation_deg(edges)
        labeled_corners = cls.corner_display_labels(corners)
        vertex_angles = cls.vertex_angles_from_edges(labeled_corners, edges)
        return {
            "corners": labeled_corners,
            "edges": edges,
            "vertex_angles": vertex_angles,
            "orientation_deg": orientation_deg,
        }

    @staticmethod
    def edge_vector_from_corner(edge, corner_pt):
        p1 = np.array(edge["p1"], dtype=np.float32)
        p2 = np.array(edge["p2"], dtype=np.float32)
        corner = np.array(corner_pt, dtype=np.float32)
        if float(np.linalg.norm(p1 - corner)) <= float(np.linalg.norm(p2 - corner)):
            return p2 - p1
        return p1 - p2

    @classmethod
    def vertex_angles_from_edges(cls, labeled_corners, edges):
        corner_pts = {label: pt for label, pt in labeled_corners}
        edge_map = {edge["label"]: edge for edge in edges}
        meets = {
            "TL": ("top", "left"),
            "TR": ("top", "right"),
            "BR": ("bottom", "right"),
            "BL": ("bottom", "left"),
        }

        vertex_angles = []
        for label in ("TL", "TR", "BR", "BL"):
            edge_a_name, edge_b_name = meets[label]
            edge_a = edge_map.get(edge_a_name)
            edge_b = edge_map.get(edge_b_name)
            corner_pt = corner_pts.get(label)
            if edge_a is None or edge_b is None or corner_pt is None:
                continue

            v1 = cls.edge_vector_from_corner(edge_a, corner_pt)
            v2 = cls.edge_vector_from_corner(edge_b, corner_pt)
            dot = float(v1[0] * v2[0] + v1[1] * v2[1])
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            angle_deg = float(math.degrees(math.atan2(abs(cross), dot)))
            vertex_angles.append((label, angle_deg))

        return vertex_angles

    @staticmethod
    def format_corners(corners):
        if not corners:
            return "none"
        return "  ".join(
            f"{label}=({x:.0f},{y:.0f})"
            for label, (x, y) in corners
        )

    @staticmethod
    def format_edges(edges):
        if not edges:
            return "none"
        return "  ".join(
            f"{edge['label']}={edge['angle_deg']:+.1f}deg ({edge['length_px']:.0f}px)"
            for edge in edges
        )

    @staticmethod
    def format_vertex_angles(vertex_angles):
        if not vertex_angles:
            return "none"
        return "  ".join(
            f"{label}={angle_deg:.1f}deg (dev={angle_deg - 90.0:+.1f})"
            for label, angle_deg in vertex_angles
        )