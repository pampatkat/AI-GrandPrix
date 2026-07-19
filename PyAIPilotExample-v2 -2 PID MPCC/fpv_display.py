"""Non-blocking FPV window with YOLO gate overlays."""

import queue
import threading
import time

import cv2


class FPVDisplay:
    WINDOW_NAME = "AI Grand Prix - FPV Gate Detection"
    GATE_COLOR = (0, 255, 0)
    OUTER_COLOR = (255, 180, 0)
    CENTER_COLOR = (255, 255, 255)

    def __init__(self, data, max_fps=10.0):
        self.data = data
        self.max_fps = max(1.0, float(max_fps))
        self._minimum_frame_interval = 1.0 / self.max_fps
        self._last_queued_time = 0.0
        self._frames = queue.Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(
            target=self._display_loop,
            name="fpv-display",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame_id, image):
        # The preview does not need to refresh at the detector's full 30 Hz.
        # Dropping preview frames here preserves CPU time for inference/control.
        now = time.perf_counter()
        if now - self._last_queued_time < self._minimum_frame_interval:
            return
        self._last_queued_time = now
        # draw_overlay performs the one required copy. The decoded frame is
        # otherwise read-only, so copying again here only burns memory bandwidth.
        frame = (frame_id, image)
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            self._frames.put_nowait(frame)

    def stop(self):
        self._running = False
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _display_loop(self):
        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, 640, 360)
            while self._running:
                try:
                    item = self._frames.get(timeout=0.1)
                except queue.Empty:
                    cv2.waitKey(1)
                    continue
                if item is None:
                    continue
                frame_id, image = item
                overlay = self.draw_overlay(
                    image,
                    frame_id,
                    self.data.get("gate_detection"),
                    self.data.get("gate_detector_status", "unknown"),
                    self.data.get("gate_track"),
                )
                cv2.imshow(self.WINDOW_NAME, overlay)
                cv2.waitKey(1)
        except cv2.error as exc:
            # Keep flight and inference running on headless systems.
            self.data["fpv_display_error"] = str(exc)
            print("FPV window unavailable; continuing without display.", flush=True)
        finally:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
                cv2.waitKey(1)
            except cv2.error:
                pass

    @classmethod
    def draw_overlay(
        cls, image, frame_id, gate_result, detector_status="ready", gate_track=None
    ):
        overlay = image.copy()
        height, width = overlay.shape[:2]
        image_center = (width // 2, height // 2)

        # FPV aiming reticle.
        cv2.line(overlay, (image_center[0] - 12, image_center[1]),
                 (image_center[0] + 12, image_center[1]), cls.CENTER_COLOR, 1)
        cv2.line(overlay, (image_center[0], image_center[1] - 12),
                 (image_center[0], image_center[1] + 12), cls.CENTER_COLOR, 1)

        gate_result = gate_result or {}
        age = time.time() - gate_result.get("time", 0.0)
        detections = gate_result.get("detections", []) if age <= 1.0 else []
        roi = gate_result.get("roi") or {}
        if roi.get("used"):
            cv2.rectangle(
                overlay,
                (int(roi["x1"]), int(roi["y1"])),
                (int(roi["x2"]), int(roi["y2"])),
                (255, 0, 255),
                1,
            )
        for index, detection in enumerate(detections):
            box = detection["bbox"]
            x1, y1 = int(box["x1"]), int(box["y1"])
            x2, y2 = int(box["x2"]), int(box["y2"])
            color = cls.GATE_COLOR if index == 0 else (0, 190, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"Gate {detection['confidence']:.2f}"
            cv2.putText(overlay, label, (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            points = []
            for point in detection.get("keypoints", []):
                confidence = point.get("confidence")
                if confidence is not None and confidence < 0.20:
                    points.append(None)
                    continue
                xy = (int(point["x"]), int(point["y"]))
                points.append(xy)
                point_color = cls.OUTER_COLOR if "outer" in point["name"] else color
                cv2.circle(overlay, xy, 4, point_color, -1, cv2.LINE_AA)

            cls._draw_polygon(overlay, points, (0, 1, 3, 2), color)
            cls._draw_polygon(overlay, points, (4, 5, 7, 6), cls.OUTER_COLOR)
            center = (int(box["center_x"]), int(box["center_y"]))
            cv2.circle(overlay, center, 6, color, 2, cv2.LINE_AA)
            if index == 0:
                cv2.line(overlay, image_center, center, (255, 0, 255), 2, cv2.LINE_AA)

        detected = bool(detections)
        if detector_status != "ready":
            status = "DETECTOR DISABLED"
            status_color = (0, 0, 255)
        else:
            status = "TRACKING GATE" if detected else "SEARCHING FOR GATE"
            status_color = cls.GATE_COLOR if detected else (0, 200, 255)
        cv2.rectangle(overlay, (0, 0), (width, 34), (20, 20, 20), -1)
        cv2.putText(overlay, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, status_color, 2, cv2.LINE_AA)
        confidence = gate_result.get("confidence", 0.0) if detected else 0.0
        provider = str(gate_result.get("execution_provider", ""))
        runtime = (
            "CUDA"
            if provider == "CUDAExecutionProvider"
            else str(gate_result.get("runtime", "--")).upper()
        )
        inference_ms = gate_result.get("inference_ms")
        timing = "--" if inference_ms is None else f"{inference_ms:.1f} ms"
        region = "ROI" if roi.get("used") else "FULL"
        info = (
            f"FPV {frame_id}  detections {len(detections)}  confidence {confidence:.2f}  "
            f"{runtime}/{region} {timing}"
        )
        cv2.putText(overlay, info, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, cls.CENTER_COLOR, 1, cv2.LINE_AA)
        pose = gate_result.get("pose") or {}
        track = gate_track or {}
        if detected and pose.get("valid"):
            pose_text = (
                f"PnP {pose['distance_m']:.2f} m  "
                f"bearing {pose['bearing_body_deg']['horizontal']:+.1f} deg  "
                f"reproj {pose['reprojection_error_px']:.1f} px"
            )
            cv2.putText(overlay, pose_text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, cls.GATE_COLOR, 1, cv2.LINE_AA)
        elif track.get("valid"):
            track_text = (
                f"EKF coast {track['distance_m']:.2f} m  "
                f"age {track['measurement_age_s']:.2f} s"
            )
            cv2.putText(overlay, track_text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (0, 200, 255), 1, cv2.LINE_AA)
        return overlay

    @staticmethod
    def _draw_polygon(image, points, indices, color):
        polygon = [points[index] for index in indices if index < len(points)]
        if len(polygon) != len(indices) or any(point is None for point in polygon):
            return
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            cv2.line(image, start, end, color, 2, cv2.LINE_AA)
