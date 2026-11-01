import cv2
import sys
import os
import time
import subprocess
import threading
import urllib.request
import numpy as np
from ultralytics import YOLO
from distance import DistanceEstimator
from rover_control import RoverController
import serial.tools.list_ports

class VideoStreamCapture:
    """
    Universal Zero-Lag Video Stream Capture:
    - Automatically manages ESP32-CAM MJPEG HTTP streams with direct socket parsing.
    - Eliminates OpenCV VideoCapture buffering lag and dropped frame bugs.
    - Seamlessly supports USB Webcams with DirectShow acceleration.
    - Supports dynamic on-the-fly source switching (Webcam <-> ESP32-CAM).
    """
    def __init__(self, source):
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        self.status_msg = "Initializing..."
        self.cap = None
        self.worker_thread = None
        self._last_wifi_check = 0.0
        self.source = None
        self.is_http = False
        self.set_source(source)

    def set_source(self, source):
        with self.lock:
            # Release previous webcam if any
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.latest_frame = None

        self.source = source
        is_webcam = False
        if isinstance(source, int):
            is_webcam = True
            cam_idx = source
        elif isinstance(source, str) and source.isdigit():
            is_webcam = True
            cam_idx = int(source)
        elif isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
            is_webcam = False
        else:
            is_webcam = True
            cam_idx = 0

        self.is_http = not is_webcam

        if self.is_http:
            self.status_msg = f"Connecting to {self.source}..."
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.worker_thread = threading.Thread(target=self._mjpeg_worker, daemon=True)
                self.worker_thread.start()
        else:
            self.status_msg = f"Opening Webcam {cam_idx}..."
            # On Windows, DirectShow provides instant startup without 10s delay
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                self.status_msg = f"Webcam {cam_idx} Active"
            else:
                self.status_msg = f"Failed to open Webcam {cam_idx}"

    def _ensure_wifi_connected(self):
        now = time.time()
        if now - self._last_wifi_check < 12.0:
            return
        self._last_wifi_check = now
        try:
            output = subprocess.check_output("netsh wlan show interfaces", shell=True, text=True)
            if "State" in output and "connected" in output and "RoboForensic_CAM" in output:
                return
            self.status_msg = "Reconnecting to RoboForensic_CAM Wi-Fi..."
            subprocess.run('netsh wlan connect name="RoboForensic_CAM"', shell=True, capture_output=True)
        except Exception:
            pass

    def _mjpeg_worker(self):
        while self.running:
            if not self.is_http:
                time.sleep(0.1)
                continue

            target_source = self.source
            if "192.168.4.1" in str(target_source):
                self._ensure_wifi_connected()

            try:
                self.status_msg = f"Connecting to stream: {target_source}..."
                req = urllib.request.Request(target_source, headers={'User-Agent': 'RoboForensic/2.0'})
                stream = urllib.request.urlopen(req, timeout=4.0)
                self.status_msg = "ESP32-CAM Stream Connected"
                buf = b''

                while self.running and self.is_http and self.source == target_source:
                    chunk = stream.read(4096)
                    if not chunk:
                        time.sleep(0.005)
                        continue
                    buf += chunk
                    a = buf.find(b'\xff\xd8')
                    if a != -1:
                        if a > 0:
                            buf = buf[a:]
                            a = 0
                        b = buf.find(b'\xff\xd9', 2)
                        if b != -1:
                            jpg = buf[:b+2]
                            buf = buf[b+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self.lock:
                                    self.latest_frame = frame
                                    self.status_msg = "Live Stream Active"
            except Exception as e:
                self.status_msg = f"Stream offline ({type(e).__name__}). Retrying..."
                time.sleep(1.0)

    def read(self):
        if self.is_http:
            with self.lock:
                if self.latest_frame is not None:
                    return True, self.latest_frame.copy()
            return False, None
        else:
            if self.cap and self.cap.isOpened():
                return self.cap.read()
            return False, None

    def release(self):
        self.running = False
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

def auto_detect_serial_port():
    """Scans for available serial ports and returns the best candidate (e.g. CH340 / COM21)."""
    try:
        ports = serial.tools.list_ports.comports()
        if not ports:
            return None
        for p in ports:
            if p.device == "COM21" or "CH340" in p.description or "USB-SERIAL" in p.description or "ESP32" in p.description:
                return p.device
        return ports[0].device
    except Exception:
        return None

def run_pipeline(stream_url="http://192.168.4.1/stream", serial_port=None, conf_threshold=0.35):
    """
    Main RoboForensic AI Engine Pipeline:
    1. Reads stream frames from ESP32-CAM or Webcam with zero lag.
    2. Runs YOLO red_object detector.
    3. Calculates target center & apparent bounding box pixel width.
    4. Computes monocular distance estimation (cm).
    5. Evaluates zone position (LEFT / CENTER / RIGHT) & determines rover command (F/L/R/S).
    6. Renders live HUD UI overlays.
    7. Sends motor command to ESP32 over serial.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "../models/best.pt")

    # Fallback to base model if custom best.pt doesn't exist yet
    if not os.path.exists(model_path):
        fallback_weights = os.path.join(script_dir, "../../yolo11n.pt")
        if os.path.exists(fallback_weights):
            model_path = fallback_weights
        else:
            model_path = "yolo11n.pt"

    # Auto-detect serial port if not provided
    if not serial_port:
        detected_port = auto_detect_serial_port()
        if detected_port:
            serial_port = detected_port
            print(f"[Main] Auto-detected Serial Port: {serial_port}")

    print("==================================================")
    print(" RoboForensic Autonomous Rover Vision Engine ")
    print("==================================================")
    print(f" Camera Stream : {stream_url}")
    print(f" YOLO Model    : {model_path}")
    print(f" Serial Port   : {serial_port if serial_port else 'Simulated / Disconnected'}")
    print("--------------------------------------------------")

    cap = VideoStreamCapture(stream_url)
    detector = YOLO(model_path)
    distance_estimator = DistanceEstimator(config_path=os.path.join(script_dir, "calibration.json"))
    controller = RoverController(port=serial_port, frame_width=640, stop_distance_cm=25.0)

    print("\nStarting live control loop...")
    print(" Hotkeys: [W] Webcam 0 | [E] ESP32-CAM Stream | [S] Stop | [Q] Quit\n")

    last_command_time = 0
    command_interval = 0.2  # Transmit commands every 200ms
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0.0
    standby_dots = 0

    try:
        while True:
            ret, frame = cap.read()

            # Responsive Standby Screen if camera is offline / reconnecting
            if not ret or frame is None:
                # SAFETY: Keep rover stopped while video feed is unavailable
                controller.send_command("S")

                standby = np.zeros((480, 640, 3), dtype=np.uint8)
                standby_dots = (standby_dots + 1) % 4
                dots = "." * standby_dots

                # Render Standby HUD
                cv2.putText(standby, "ROBOFORENSIC VISION ENGINE", (110, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(standby, f"STATUS: {cap.status_msg}{dots}", (50, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(standby, f"Source: {cap.source}", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

                serial_status = f"Connected ({serial_port})" if (controller.serial_conn and controller.serial_conn.is_open) else "Simulated / Disconnected"
                s_color = (0, 255, 0) if (controller.serial_conn and controller.serial_conn.is_open) else (150, 150, 150)
                cv2.putText(standby, f"Serial: {serial_status}", (50, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1, cv2.LINE_AA)

                # Hotkeys Help Box
                cv2.rectangle(standby, (30, 330), (610, 420), (35, 35, 35), -1)
                cv2.putText(standby, "AVAILABLE COMMANDS:", (50, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(standby, "[W] Switch to Webcam 0   |   [E] Switch to ESP32-CAM   |   [Q] Quit",
                            (50, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

                cv2.imshow("RoboForensic Vision & Navigation HUD", standby)
                key = cv2.waitKey(60) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('w') or key == ord('W'):
                    print("[Main] User switched to Local Webcam (0)")
                    cap.set_source(0)
                elif key == ord('e') or key == ord('E'):
                    print("[Main] User switched to ESP32-CAM Stream")
                    cap.set_source("http://192.168.4.1/stream")
                continue

            # Standardize frame resolution
            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape

            # FPS calculation
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # YOLO Object Detection (imgsz=320 for ultra-fast, zero-lag inference)
            results = detector(frame, imgsz=320, conf=conf_threshold, verbose=False)[0]

            best_box = None
            max_area = 0

            # Find largest detected target
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2, float(box.conf[0]))

            target_center_x = None
            target_center_y = None
            distance_cm = None
            command_code = "SEARCH"
            zone_label = "NO TARGET"
            description = "Searching for target..."

            if best_box:
                x1, y1, x2, y2, conf = best_box
                box_width = x2 - x1
                target_center_x = (x1 + x2) / 2.0
                target_center_y = (y1 + y2) / 2.0

                # Monocular Distance Estimation
                distance_cm = distance_estimator.estimate_distance(box_width, smooth=True)

                # Determine Rover Action Command
                command_code, zone_label, description = controller.determine_action(target_center_x, distance_cm)

                # Draw Target Bounding Box & Center Marker
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.circle(frame, (int(target_center_x), int(target_center_y)), 6, (0, 255, 0), -1)

                # Bounding Box Label
                lbl = f"Target {conf*100:.0f}% | W:{int(box_width)}px"
                cv2.putText(frame, lbl, (int(x1), max(15, int(y1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                distance_estimator.reset_history()

            # Transmit command periodically
            if now - last_command_time >= command_interval:
                controller.send_command(command_code)
                last_command_time = now

            # RENDER HUD UI OVERLAYS
            # Zone Lines
            left_line = controller.left_threshold
            right_line = controller.right_threshold
            cv2.line(frame, (left_line, 0), (left_line, h), (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(frame, (right_line, 0), (right_line, h), (200, 200, 200), 1, cv2.LINE_AA)

            # Zone Titles
            cv2.putText(frame, "LEFT", (left_line // 3, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(frame, "CENTER", (left_line + 45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(frame, "RIGHT", (right_line + 45, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            # Top Telemetry Header Bar
            header = frame.copy()
            cv2.rectangle(header, (0, 0), (w, 35), (0, 0, 0), -1)
            cv2.addWeighted(header, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            src_text = "ESP32-CAM" if cap.is_http else "WEBCAM"
            cv2.putText(frame, f"SRC: {src_text} ([W]/[E])", (140, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            serial_status = f"SERIAL: {serial_port}" if (controller.serial_conn and controller.serial_conn.is_open) else "SERIAL: DISCONNECTED"
            s_color = (0, 255, 0) if (controller.serial_conn and controller.serial_conn.is_open) else (120, 120, 255)
            cv2.putText(frame, serial_status, (420, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 1)

            # Bottom Overlay HUD Box
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, h - 105), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # HUD Metrics
            dist_str = f"{distance_cm:.1f} cm" if distance_cm is not None else "---"
            cmd_color = (0, 255, 0) if command_code == "F" else (0, 0, 255) if command_code == "S" else (255, 255, 0)

            cv2.putText(frame, f"ZONE: {zone_label}", (25, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"DISTANCE: {dist_str}", (210, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"COMMAND: {command_code}", (450, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cmd_color, 3)
            cv2.putText(frame, description, (25, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 250, 200), 1)

            cv2.imshow("RoboForensic Vision & Navigation HUD", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('w') or key == ord('W'):
                print("[Main] Switching to Local Webcam 0...")
                cap.set_source(0)
            elif key == ord('e') or key == ord('E'):
                print("[Main] Switching to ESP32-CAM stream...")
                cap.set_source("http://192.168.4.1/stream")
            elif key == ord('s') or key == ord('S'):
                controller.send_command("S")
                print("[Main] Emergency STOP executed.")

    finally:
        try:
            controller.send_command("S")
        except Exception:
            pass
        cap.release()
        controller.close()
        cv2.destroyAllWindows()
        print("[Main] Pipeline terminated cleanly.")

if __name__ == "__main__":
    stream_url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.4.1/stream"
    serial_port = sys.argv[2] if len(sys.argv) > 2 else None
    run_pipeline(stream_url=stream_url, serial_port=serial_port)
