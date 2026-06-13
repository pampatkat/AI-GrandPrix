import socket
import struct
import threading
import time

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
VISION_DEBUG_INTERVAL_S = 0.5

# HSV range for the simulator's orange gate. Keep the hue range broad enough for
# JPEG compression and lighting changes, but require saturation so white/gray
# track elements are ignored.
ORANGE_HSV_LOWER = np.array([3, 80, 70], dtype=np.uint8)
ORANGE_HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)

class VisionRX:

    def __init__(self, data):
        self.data = data
        self.last_vision_debug_print_time = 0.0
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
        self.data["gate_detection"] = detection
        self.print_gate_debug(detection)

    def detect_gate(self, img):
        height, width = img.shape[:2]
        orange_candidates = self.find_orange_gate_candidates(img)
        gate, best_rejected_score = self.choose_orange_gate_candidate(
            orange_candidates,
            width,
            height
        )

        square_candidates = []
        if gate is None:
            square_candidates = self.find_square_candidates(img)
            gate, best_rejected_score = self.choose_square_gate_candidate(
                square_candidates,
                width,
                height
            )

        if gate is None:
            return {
                "detected": False,
                "confidence": 0.0,
                "orange_candidates": len(orange_candidates),
                "square_candidates": len(square_candidates),
                "best_rejected_score": best_rejected_score,
            }

        center_x, center_y = gate["center"]
        size_px = gate["size_px"]
        return {
            "detected": True,
            "confidence": gate["confidence"],
            "match_type": gate["match_type"],
            "orange_candidates": len(orange_candidates),
            "square_candidates": len(square_candidates),
            "best_rejected_score": best_rejected_score,
            "center_x": center_x,
            "center_y": center_y,
            "offset_x": (center_x - (width * 0.5)) / (width * 0.5),
            "offset_y": (center_y - (height * 0.5)) / (height * 0.5),
            "size_px": size_px,
            "size_ratio": size_px / max(1.0, min(width, height)),
            "image_width": width,
            "image_height": height,
            "inner_clearance_m": min(
                GATE_INNER_WIDTH_M - DRONE_WIDTH_M,
                GATE_INNER_HEIGHT_M - DRONE_HEIGHT_M,
            ),
        }

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
            if aspect > 1.75:
                continue

            fill_ratio = orange_pixels / rect_area
            # The physical gate is an orange square frame: orange area is roughly
            # outer square minus inner square, about 69% of the outer footprint.
            if fill_ratio < 0.18 or fill_ratio > 0.92:
                continue

            center = rect[0]
            candidates.append({
                "center": center,
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
            print(
                "Vision gate "
                f"detected conf={detection.get('confidence', 0.0):.2f} "
                f"match={detection.get('match_type', 'unknown')} "
                f"orange={detection.get('orange_candidates', 0)} "
                f"squares={detection.get('square_candidates', 0)} "
                f"offset=({detection.get('offset_x', 0.0):+.2f}, {detection.get('offset_y', 0.0):+.2f}) "
                f"size={detection.get('size_ratio', 0.0):.2f}",
                flush=True
            )
        else:
            print(
                "Vision gate none "
                f"orange={detection.get('orange_candidates', 0)} "
                f"squares={detection.get('square_candidates', 0)} "
                f"best_rejected_score={detection.get('best_rejected_score', 0.0):.2f}",
                flush=True
            )

    @staticmethod
    def normalized_center_error(center_a, center_b, scale):
        dx = center_a[0] - center_b[0]
        dy = center_a[1] - center_b[1]
        return float(np.hypot(dx, dy) / max(1.0, scale))