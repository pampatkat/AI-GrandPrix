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
# How long the last good detection is "coasted" after a frame fails to detect,
# so a few borderline misses don't blink the overlay off.
GATE_DETECTION_HOLD_S = 0.60
# Detection hysteresis: a frame's (smoothed) confidence must reach
# GATE_DETECT_ON_CONFIDENCE to START reporting a gate, but only has to stay above
# the lower GATE_DETECT_OFF_CONFIDENCE to KEEP reporting it. This stops the
# on/off chatter when the raw score hovers right around a single threshold (e.g.
# a far/angled gate jittering from per-frame JPEG/render noise on a still drone).
GATE_DETECT_ON_CONFIDENCE = 0.45
GATE_DETECT_OFF_CONFIDENCE = 0.30
# EMA factor used to smooth the per-frame confidence before the on/off decision.
# Lower = more smoothing (steadier, slower to react); higher = snappier.
GATE_DETECTION_CONF_ALPHA = 0.45
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

class VisionRX:

    def __init__(self, data):
        self.data = data
        self.last_vision_debug_print_time = 0.0
        self._overlay_window_ready = False
        self._metric_history = deque(maxlen=METRIC_HISTORY_LEN)
        self._last_gate_detection = None
        # Hysteresis state for the on/off decision (see stabilize_gate_detection).
        self._smoothed_confidence = None
        self._detection_active = False
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
            edges = detection.get("edges") or []
            outer_corners = detection.get("outer_corners")
            inner_corners = detection.get("inner_corners")
            outer_edges = detection.get("outer_edges") or []
            inner_edges = detection.get("inner_edges") or []
            outer_corner_colors = {
                "TL": (0, 255, 255),
                "TR": (0, 200, 255),
                "BR": (0, 140, 255),
                "BL": (0, 80, 255),
            }
            outer_edge_colors = {
                "top": (0, 255, 0),
                "right": (255, 180, 0),
                "bottom": (0, 180, 255),
                "left": (255, 0, 180),
            }
            inner_color = (120, 255, 120)

            # Both detections run every frame: draw the outer gate corners (orange frame)
            # and the inner gate corners (fly-through aperture) at the same time.
            if outer_corners:
                self._draw_gate_quad(
                    overlay, outer_corners, outer_edges,
                    outer_corner_colors, outer_edge_colors,
                )
            if inner_corners:
                self._draw_gate_quad(
                    overlay, inner_corners, inner_edges,
                    inner_color, inner_color, label_prefix="i:",
                )
            if not outer_corners and not inner_corners:
                self._draw_gate_quad(
                    overlay, detection.get("corners") or [], edges,
                    outer_corner_colors, outer_edge_colors,
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

        self._draw_gate_candidate_markers(overlay, detection)
        self._draw_detection_graph_panel(overlay, detection)
        self._draw_pnp_graph_panel(overlay, detection)
        self._draw_gate_attitude_panel(overlay, detection)
        return overlay

    @staticmethod
    def _draw_gate_candidate_markers(overlay, detection):
        # Annotate every gate candidate (nearest first) with its estimated
        # distance so all gates in view are labeled, not just the primary one.
        candidates = detection.get("gate_candidates") or []
        if not candidates:
            return
        primary_x = detection.get("center_x")
        primary_y = detection.get("center_y")
        for idx, cand in enumerate(candidates):
            cx = int(round(cand["center_x"]))
            cy = int(round(cand["center_y"]))
            is_primary = (
                primary_x is not None
                and abs(cand["center_x"] - float(primary_x)) < 2.0
                and abs(cand["center_y"] - float(primary_y)) < 2.0
            )
            # Nearest candidate (or the primary gate) is highlighted green; the
            # rest are dimmer cyan so the scene reads as "this one, then those".
            color = (0, 255, 0) if (is_primary or idx == 0) else (200, 200, 80)
            cv2.circle(overlay, (cx, cy), 9, color, 1, lineType=cv2.LINE_AA)
            cv2.drawMarker(
                overlay, (cx, cy), color,
                markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=1,
            )
            dist = cand.get("distance_m")
            dist_text = f"{dist:.1f}m" if dist is not None else "n/a"
            cv2.putText(
                overlay,
                f"#{idx + 1} {dist_text}",
                (cx + 11, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_gate_quad(overlay, corners, edges, corner_colors, edge_colors, label_prefix=""):
        def _color(palette, key, fallback):
            if isinstance(palette, dict):
                return palette.get(key, fallback)
            return palette

        for edge in edges:
            p1 = tuple(int(round(v)) for v in edge["p1"])
            p2 = tuple(int(round(v)) for v in edge["p2"])
            color = _color(edge_colors, edge.get("label"), (0, 255, 0))
            cv2.line(overlay, p1, p2, color, 2, cv2.LINE_AA)

        for label, point in corners:
            pt = (int(round(point[0])), int(round(point[1])))
            color = _color(corner_colors, label, (0, 255, 255))
            cv2.circle(overlay, pt, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"{label_prefix}{label}",
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

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

    def _draw_gate_attitude_panel(self, overlay, detection):
        """Top-right diagram: the gate drawn tilted at its detected degree, with a
        drone icon showing where the drone sits relative to the gate opening."""
        height, width = overlay.shape[:2]
        panel_w = 200
        panel_h = 180
        x0 = width - panel_w - 12
        y0 = 12
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)
        cv2.putText(
            overlay,
            "Gate & drone",
            (x0 + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        cx = x0 + panel_w // 2
        cy = y0 + panel_h // 2 + 14

        if not detection.get("detected"):
            cv2.putText(
                overlay,
                "no gate",
                (x0 + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )
            return

        orientation_deg = float(detection.get("orientation_deg", 0.0))
        offset_x = float(detection.get("offset_x", 0.0))
        offset_y = float(detection.get("offset_y", 0.0))
        if detection.get("pnp_valid"):
            orientation_deg = float(detection.get("pnp_orientation_deg", orientation_deg))
            offset_x = float(detection.get("pnp_offset_x", offset_x))
            offset_y = float(detection.get("pnp_offset_y", offset_y))

        # Horizon reference (level) so the gate's tilt is read against it.
        cv2.line(
            overlay,
            (x0 + 14, cy),
            (x0 + panel_w - 14, cy),
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

        # Gate square, rotated by its detected degree and shifted by the offset so
        # the diagram reflects where the opening sits relative to the drone.
        gate_size = 84.0
        half = gate_size * 0.5
        angle_rad = math.radians(orientation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        gx = cx + int(round(offset_x * 60.0))
        gy = cy + int(round(offset_y * 60.0))
        local = [(-half, -half), (half, -half), (half, half), (-half, half)]
        pts = []
        for lx, ly in local:
            px = gx + lx * cos_a - ly * sin_a
            py = gy + lx * sin_a + ly * cos_a
            pts.append((int(round(px)), int(round(py))))
        for i in range(4):
            cv2.line(overlay, pts[i], pts[(i + 1) % 4], (0, 200, 255), 2, cv2.LINE_AA)
        # Inner opening hint.
        inner = []
        for lx, ly in [(p * 0.55, q * 0.55) for p, q in local]:
            px = gx + lx * cos_a - ly * sin_a
            py = gy + lx * sin_a + ly * cos_a
            inner.append((int(round(px)), int(round(py))))
        for i in range(4):
            cv2.line(overlay, inner[i], inner[(i + 1) % 4], (0, 140, 200), 1, cv2.LINE_AA)

        # Tilt arc + degree label, drawn from the level horizon up to the gate's
        # top edge so the angle is obvious at a glance.
        arc_r = 34
        if orientation_deg >= 0:
            start_ang, end_ang = 0, int(round(orientation_deg))
        else:
            start_ang, end_ang = int(round(orientation_deg)), 0
        cv2.ellipse(
            overlay, (gx, gy), (arc_r, arc_r), 0,
            start_ang, end_ang, (0, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.line(
            overlay, (gx, gy),
            (gx + int(round(arc_r * cos_a)), gy + int(round(arc_r * sin_a))),
            (0, 255, 255), 1, cv2.LINE_AA,
        )

        # Drone icon: a small level quad (X-frame with rotors) at the drone's view
        # center, i.e. where the camera is pointed.
        self._draw_drone_icon(overlay, cx, cy)
        cv2.line(overlay, (cx, cy), (gx, gy), (180, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(
            overlay,
            f"gate {orientation_deg:+.1f} deg",
            (x0 + 8, y0 + panel_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_drone_icon(overlay, cx, cy, arm=11):
        # Simple top/front-view quad: two diagonal arms with a rotor disk at each
        # tip and a body dot in the middle.
        tips = [
            (cx - arm, cy - arm),
            (cx + arm, cy - arm),
            (cx + arm, cy + arm),
            (cx - arm, cy + arm),
        ]
        cv2.line(overlay, tips[0], tips[2], (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, tips[1], tips[3], (255, 255, 255), 1, cv2.LINE_AA)
        for tip in tips:
            cv2.circle(overlay, tip, 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 2, (80, 180, 255), -1, lineType=cv2.LINE_AA)

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

        # Smooth the raw per-frame confidence so JPEG/render noise can't make the
        # decision chatter, then apply hysteresis: a strong score is needed to turn
        # the gate ON, but only a weak score to keep it ON. This is what stops the
        # overlay blinking on/off while the drone is sitting still.
        if detection.get("detected"):
            raw_confidence = float(detection.get("confidence", 0.0))
            if self._smoothed_confidence is None:
                self._smoothed_confidence = raw_confidence
            else:
                self._smoothed_confidence += (
                    GATE_DETECTION_CONF_ALPHA
                    * (raw_confidence - self._smoothed_confidence)
                )
        elif self._smoothed_confidence is not None:
            self._smoothed_confidence *= (1.0 - GATE_DETECTION_CONF_ALPHA)

        smoothed_confidence = self._smoothed_confidence or 0.0
        if detection.get("detected"):
            threshold = (
                GATE_DETECT_OFF_CONFIDENCE
                if self._detection_active
                else GATE_DETECT_ON_CONFIDENCE
            )
            accept = smoothed_confidence >= threshold
        else:
            accept = False

        if accept:
            self._detection_active = True
            detection["raw_detected"] = True
            detection["held_detection"] = False
            detection["confidence"] = smoothed_confidence
            self._last_gate_detection = detection.copy()
            return detection

        if self._last_gate_detection is None:
            self._detection_active = False
            detection["raw_detected"] = False
            detection["held_detection"] = False
            return detection

        last_time = float(self._last_gate_detection.get("time", 0.0))
        hold_age = now - last_time
        if hold_age < 0.0 or hold_age > GATE_DETECTION_HOLD_S:
            self._last_gate_detection = None
            self._detection_active = False
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
        held["gate_candidates"] = detection.get("gate_candidates", held.get("gate_candidates", []))
        held["gate_candidate_count"] = detection.get("gate_candidate_count", held.get("gate_candidate_count", 0))
        held["confidence"] = max(0.45, float(held.get("confidence", 0.0)) * 0.9)
        return held

    def detect_gate(self, img):
        height, width = img.shape[:2]
        square_candidates = self.find_square_candidates(img)
        orange_candidates = self.find_orange_gate_candidates(img)

        # The outer (orange frame) and inner (square aperture) detections run together
        # on every frame. The inner detection is NOT a fallback for the outer one -
        # both are always computed and reported so both corner sets can be shown at the
        # same time.
        outer_gate, outer_rejected = self.choose_orange_gate_candidate(
            orange_candidates,
            width,
            height
        )
        inner_gate, inner_rejected = self.choose_square_gate_candidate(
            square_candidates,
            width,
            height
        )

        # Distance-tagged list of every gate candidate in view (nearest first),
        # kept alongside the single primary detection so far/low gates that lose
        # the primary-gate scoring contest aren't discarded.
        gate_candidates = self._build_gate_candidate_list(
            orange_candidates, width, height
        )

        outer_detection = None
        if outer_gate is not None:
            outer_corners, outer_source = self.refine_gate_corners(outer_gate, square_candidates)
            outer_detection = self._compose_gate_detection(
                outer_gate,
                outer_corners,
                outer_source,
                width,
                height,
                self.gate_outer_object_points(),
            )

        inner_detection = None
        if inner_gate is not None:
            inner_detection = self._compose_gate_detection(
                inner_gate,
                inner_gate["corners"],
                inner_gate.get("corner_source", "inner_square"),
                width,
                height,
                self.gate_inner_object_points(),
            )

        best_rejected_score = max(outer_rejected, inner_rejected)

        if outer_detection is None and inner_detection is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "orange_candidates": len(orange_candidates),
                "square_candidates": len(square_candidates),
                "best_rejected_score": best_rejected_score,
                "gate_candidates": gate_candidates,
                "gate_candidate_count": len(gate_candidates),
            }

        # The outer detection drives the top-level pose consumed by the controller and
        # HUD. When the outer frame isn't visible we surface the inner result instead so
        # the gate still counts as detected whenever either detection fires.
        detection = dict(outer_detection if outer_detection is not None else inner_detection)
        detection["orange_candidates"] = len(orange_candidates)
        detection["square_candidates"] = len(square_candidates)
        detection["best_rejected_score"] = best_rejected_score
        detection["gate_candidates"] = gate_candidates
        detection["gate_candidate_count"] = len(gate_candidates)
        detection["outer_detected"] = outer_detection is not None
        detection["inner_detected"] = inner_detection is not None
        # True when the image-center "white cross" (where the drone is pointed)
        # sits inside the inner gate aperture. The controller uses this to stop
        # rotating once we're lined up to fly straight through the opening.
        detection["inner_contains_image_center"] = False

        if outer_detection is not None:
            detection["outer_corners"] = outer_detection["corners"]
            detection["outer_edges"] = outer_detection["edges"]
            detection["outer_confidence"] = outer_detection["confidence"]
            detection["outer_center_x"] = outer_detection["center_x"]
            detection["outer_center_y"] = outer_detection["center_y"]
            detection["outer_distance_m"] = outer_detection["distance_estimate_m"]

        if inner_detection is not None:
            detection["inner_corners"] = inner_detection["corners"]
            detection["inner_edges"] = inner_detection["edges"]
            detection["inner_confidence"] = inner_detection["confidence"]
            detection["inner_center_x"] = inner_detection["center_x"]
            detection["inner_center_y"] = inner_detection["center_y"]
            detection["inner_distance_m"] = inner_detection["distance_estimate_m"]
            detection["inner_size_ratio"] = inner_detection["size_ratio"]
            detection["inner_contains_image_center"] = self._image_center_in_corners(
                inner_detection["corners"], width, height
            )
            # Where the image-center white cross sits INSIDE the inner aperture,
            # normalized by the aperture's half-size (not the image size): 0.0 =
            # cross dead-center in the opening, +/-1.0 = cross right on an inner
            # edge/corner, |value| > 1.0 = cross outside the aperture. The
            # controller uses these to push the cross back toward the middle of
            # the inner frame (away from the corners).
            recenter_x, recenter_y, corner_proximity = self._inner_aperture_recenter(
                inner_detection["corners"],
                inner_detection["center_x"],
                inner_detection["center_y"],
                width,
                height,
            )
            detection["inner_recenter_x"] = recenter_x
            detection["inner_recenter_y"] = recenter_y
            detection["inner_corner_proximity"] = corner_proximity

        return detection

    @staticmethod
    def _inner_aperture_recenter(corners, center_x, center_y, width, height):
        # Signed offset of the inner-aperture center from the image-center white
        # cross, normalized by the aperture half-size so the signal stays
        # meaningful as the gate grows in the frame on approach.
        #   recenter_x > 0  -> aperture center is right of the white cross
        #   recenter_y > 0  -> aperture center is below the white cross
        # Driving these to zero parks the white cross in the middle of the inner
        # frame, which is the point furthest from every inner corner.
        if not corners:
            return 0.0, 0.0, 0.0
        xs = [float(pt[0]) for _, pt in corners]
        ys = [float(pt[1]) for _, pt in corners]
        half_w = max(1.0, (max(xs) - min(xs)) * 0.5)
        half_h = max(1.0, (max(ys) - min(ys)) * 0.5)
        recenter_x = (float(center_x) - width * 0.5) / half_w
        recenter_y = (float(center_y) - height * 0.5) / half_h
        corner_proximity = min(2.0, max(abs(recenter_x), abs(recenter_y)))
        return recenter_x, recenter_y, corner_proximity

    @staticmethod
    def _image_center_in_corners(corners, width, height):
        # Point-in-polygon test: is the image center (the white cross) inside the
        # quad described by `corners` (a list of (label, (x, y)) pixel points)?
        if not corners or len(corners) < 3:
            return False
        pts = np.array(
            [[float(pt[0]), float(pt[1])] for _, pt in corners],
            dtype=np.float32,
        )
        center = (float(width) * 0.5, float(height) * 0.5)
        return cv2.pointPolygonTest(pts, center, False) >= 0

    def _compose_gate_detection(self, gate, geometry_corners, corner_source, width, height, object_points):
        perspective_asymmetry = self.corner_perspective_asymmetry(geometry_corners)
        if FRONT_FACING_GATE_CORRECTION:
            geometry_corners = self.front_facing_square_corners(geometry_corners)
            corner_source = f"{corner_source}+front_square"
        center_x, center_y = gate["center"]
        size_px = gate["size_px"]
        size_ratio = size_px / max(1.0, min(width, height))
        geometry = self.gate_geometry_from_corners(geometry_corners)
        pnp_pose = self.estimate_gate_pose_pnp(geometry["corners"], width, height, object_points)
        distance_estimate_m = self.estimate_gate_distance_m(size_ratio)
        if pnp_pose.get("pnp_valid"):
            distance_estimate_m = pnp_pose["pnp_distance_m"]
        return {
            "detected": True,
            "confidence": gate["confidence"],
            "match_type": gate["match_type"],
            "corner_source": corner_source,
            "perspective_asymmetry": perspective_asymmetry,
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

    def _build_gate_candidate_list(self, candidates, image_width, image_height):
        # Build a distance-tagged entry for EVERY orange gate candidate in the
        # frame (not just the single best one the controller flies toward), so a
        # far/low gate caught only as a glimpse in a corner isn't thrown away.
        # The score mirrors choose_orange_gate_candidate so the per-candidate
        # confidence here matches how the primary gate would be ranked. Entries
        # are sorted nearest-first (unknown distance sorts last).
        image_area = max(1.0, float(image_width * image_height))
        image_center = (image_width * 0.5, image_height * 0.5)
        half_w = max(1.0, image_width * 0.5)
        half_h = max(1.0, image_height * 0.5)
        smallest_dim = max(1.0, float(min(image_width, image_height)))

        entries = []
        for candidate in candidates:
            aspect_error = abs(candidate["aspect"] - 1.0)
            fill_error = abs(candidate["fill_ratio"] - GATE_FRAME_TO_OUTER_AREA)
            center_error = self.normalized_center_error(
                candidate["center"],
                image_center,
                smallest_dim,
            )
            area_score = min(0.25, candidate["rect_area"] / image_area)
            score = (
                1.0
                - min(1.0, aspect_error / 0.75) * 0.30
                - min(1.0, fill_error / 0.35) * 0.30
                - min(1.0, center_error / 0.75) * 0.20
                + area_score
            )
            center_x, center_y = candidate["center"]
            size_px = float(candidate["size_px"])
            size_ratio = size_px / smallest_dim
            entries.append({
                "center_x": float(center_x),
                "center_y": float(center_y),
                "offset_x": (float(center_x) - half_w) / half_w,
                "offset_y": (float(center_y) - half_h) / half_h,
                "size_px": size_px,
                "size_ratio": size_ratio,
                "distance_m": self.estimate_gate_distance_m(size_ratio),
                "confidence": max(0.0, min(1.0, score)),
                "aspect": float(candidate["aspect"]),
                "fill_ratio": float(candidate["fill_ratio"]),
            })

        entries.sort(
            key=lambda e: e["distance_m"] if e["distance_m"] is not None else float("inf")
        )
        return entries

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

        # Keep candidates down to the lower hysteresis floor; stabilize_gate_detection
        # applies the ON/OFF thresholds so the decision doesn't chatter at 0.45.
        if best_gate is not None and best_gate["confidence"] >= GATE_DETECT_OFF_CONFIDENCE:
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
                        "corners": inner["corners"],
                        "outer_corners": outer["corners"],
                        "corner_source": "inner_square",
                        "size_px": inner["size_px"],
                        "confidence": max(0.0, min(1.0, score)),
                        "match_type": "inner_outer",
                    }

        if best_gate is not None and best_gate["confidence"] >= GATE_DETECT_OFF_CONFIDENCE:
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
                f"squares={detection.get('square_candidates', 0)}\n"
                f"  gates   : {self.format_gate_candidates(detection.get('gate_candidates'))}",
                flush=True
            )
        else:
            print(
                "\n[Vision]\n"
                "  gate    : none\n"
                f"  samples : orange={detection.get('orange_candidates', 0)}  "
                f"squares={detection.get('square_candidates', 0)}  "
                f"best_rejected_score={detection.get('best_rejected_score', 0.0):.2f}\n"
                f"  gates   : {self.format_gate_candidates(detection.get('gate_candidates'))}",
                flush=True
            )

    @staticmethod
    def format_gate_candidates(candidates):
        if not candidates:
            return "none"
        parts = []
        for idx, cand in enumerate(candidates):
            dist = cand.get("distance_m")
            dist_text = f"{dist:.1f}m" if dist is not None else "n/a"
            parts.append(
                f"#{idx + 1} {dist_text} "
                f"(x={cand.get('offset_x', 0.0):+.2f},y={cand.get('offset_y', 0.0):+.2f},"
                f"c={cand.get('confidence', 0.0):.2f})"
            )
        return "  ".join(parts)

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
        pts = np.array(corners, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0)
        labeled = []
        for point in pts:
            dx = float(point[0] - center[0])
            dy = float(point[1] - center[1])
            if dy <= 0.0 and dx <= 0.0:
                label = "TL"
            elif dy <= 0.0:
                label = "TR"
            elif dx > 0.0:
                label = "BR"
            else:
                label = "BL"
            labeled.append((label, (float(point[0]), float(point[1]))))

        display_order = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
        labeled.sort(key=lambda item: display_order[item[0]])
        return labeled

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

    @classmethod
    def gate_inner_object_points(cls):
        half_w = GATE_INNER_WIDTH_M * 0.5
        half_h = GATE_INNER_HEIGHT_M * 0.5
        # Inner aperture plane uses the same axes as the outer frame.
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
    def estimate_gate_pose_pnp(cls, labeled_corners, image_width, image_height, object_points=None):
        invalid = cls._empty_pnp_pose()
        if not labeled_corners or len(labeled_corners) != 4:
            return invalid

        corner_map = {label: pt for label, pt in labeled_corners}
        required = ("TL", "TR", "BR", "BL")
        if not all(label in corner_map for label in required):
            return invalid

        image_points = np.array(
            [corner_map[label] for label in required],
            dtype=np.float64,
        )
        if object_points is None:
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