import socket
import struct
import threading

import cv2
import numpy as np

# Modify these properties if you want to run the server remotely for example
SIM_SERVER_UDP_IP = "0.0.0.0"
SIM_SERVER_UDP_PORT = 5600

class VisionRX:

    def __init__(self, data):
        self.data = data
        self.thread = threading.Thread(
            target=self._vision_loop,
            daemon=False
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
        # # Run lightweight gate detection and overlay the result on the displayed frame.
        # display_frame = img.copy()

        # # Preprocess the image for edge-based detection.
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # edges = cv2.Canny(blurred, 50, 150)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        # contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # best_gate = None
        # best_score = 0.0
        # frame_area = img.shape[0] * img.shape[1]
        # min_area = max(4000.0, 0.01 * frame_area)

        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area < min_area:
        #         continue

        #     rect = cv2.minAreaRect(contour)
        #     box = cv2.boxPoints(rect).astype(np.int32)
        #     rect_w, rect_h = rect[1]
        #     if rect_w <= 0 or rect_h <= 0:
        #         continue

        #     rect_area = rect_w * rect_h
        #     if rect_area <= 0:
        #         continue

        #     solidity = area / rect_area
        #     if solidity < 0.35:
        #         continue

        #     aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h)
        #     if aspect_ratio < 0.5 or aspect_ratio > 4.0:
        #         continue

        #     approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        #     if len(approx) < 4 or len(approx) > 8:
        #         continue

        #     score = area * solidity / aspect_ratio
        #     if score > best_score:
        #         best_score = score
        #         best_gate = {
        #             'box': box,
        #             'center': (int(rect[0][0]), int(rect[0][1])),
        #             'area': area,
        #             'aspect_ratio': aspect_ratio,
        #             'solidity': solidity,
        #         }

        # cv2.putText(
        #     display_frame,
        #     f"Frame {frame_id}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.9,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA
        # )

        # if best_gate is not None:
        #     cv2.drawContours(display_frame, [best_gate['box']], -1, (0, 255, 0), 3)
        #     cv2.circle(display_frame, best_gate['center'], 6, (0, 255, 255), -1)
        #     cv2.putText(
        #         display_frame,
        #         "Gate detected",
        #         (10, 70),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.8,
        #         (0, 255, 0),
        #         2,
        #         cv2.LINE_AA
        #     )
        #     cv2.putText(
        #         display_frame,
        #         f"A={int(best_gate['area'])} AR={best_gate['aspect_ratio']:.2f}",
        #         (10, 105),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.6,
        #         (0, 255, 0),
        #         1,
        #         cv2.LINE_AA
        #     )

        #     self.data['vision'] = {
        #         'last_frame_id': frame_id,
        #         'frame_shape': img.shape,
        #         'detected': True,
        #         'gate_center': best_gate['center'],
        #         'gate_box': best_gate['box'].tolist(),
        #         'gate_area': best_gate['area'],
        #         'gate_aspect_ratio': best_gate['aspect_ratio'],
        #         'gate_solidity': best_gate['solidity'],
        #         'gate_score': float(best_score),
        #     }
        # else:
        #     cv2.putText(
        #         display_frame,
        #         "No gate detected",
        #         (10, 70),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.8,
        #         (0, 0, 255),
        #         2,
        #         cv2.LINE_AA
        #     )
        #     self.data['vision'] = {
        #         'last_frame_id': frame_id,
        #         'frame_shape': img.shape,
        #         'detected': False,
        #     }

        # cv2.imshow("VisionRX - FPV Stream", display_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     self.is_running = False
        pass
