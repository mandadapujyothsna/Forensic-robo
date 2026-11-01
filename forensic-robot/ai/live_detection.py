"""
RoboForensic Autonomous Robot - Live Detection Module
=====================================================
Stage: ESP32-CAM -> Live Video -> YOLO -> red_object detection ->
       Bounding Box -> Confidence -> Center Coordinates -> LEFT / CENTER / RIGHT

Features:
- Connects to ESP32-CAM stream (http://192.168.4.1/stream) or custom video/webcam source.
- Loads trained YOLO model (C:\\Users\\manda\\Desktop\\forensic robot\\ai\\runs\\red_object_training\\weights\\best.pt).
- Configurable confidence threshold (default: 0.25).
- Real-time bounding box, confidence score, and center X/Y coordinate calculation.
- 3-Zone camera frame division: LEFT | CENTER | RIGHT.
- Primary target selection: Chooses the largest valid detection when multiple objects appear.
- On-screen HUD displaying target, confidence, center coordinates, zone position, or "NO RED OBJECT DETECTED".
- Clean terminal output logging detection state and telemetry.
- Safe error handling for missing model, connection failure, and frame read errors.
- Clean shutdown on 'Q', 'q', or 'ESC'.
- NOTE: No rover motor control or robotic arm activation in this stage.
"""

import os
import sys
import time
import socket
import argparse
import urllib.parse
import cv2
import numpy as np
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION DEFAULTS
# ==============================================================================
DEFAULT_MODEL_PATH = r"C:\Users\manda\Desktop\forensic robot\ai\runs\red_object_training\weights\best.pt"
FALLBACK_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forensic-robot", "ai", "models", "best.pt")
DEFAULT_STREAM_URL = "http://192.168.4.1/stream"
DEFAULT_CONF_THRESHOLD = 0.25
TARGET_CLASS_NAME = "red_object"


# ==============================================================================
# NETWORK & CONNECTION HELPERS
# ==============================================================================
def verify_model_exists(model_path: str) -> str:
    """
    Verifies that the YOLO model weights file exists.
    Falls back to alternate project weights if available.
    Exits safely with diagnostic error message if not found.
    """
    if os.path.exists(model_path):
        return model_path

    # Check local workspace fallback
    if os.path.exists(FALLBACK_MODEL_PATH):
        print(f"[WARNING] Primary model not found at: {model_path}")
        print(f"[WARNING] Falling back to local model: {FALLBACK_MODEL_PATH}")
        return FALLBACK_MODEL_PATH

    print("\n" + "=" * 70)
    print(" [ERROR] YOLO MODEL WEIGHTS FILE NOT FOUND")
    print("=" * 70)
    print(f" Checked path: {model_path}")
    print("\n Please ensure that:")
    print("  1. The training finished and generated 'best.pt'.")
    print("  2. The path to 'best.pt' is correct.")
    print("  3. You pass --model <path> if weights are stored in a different location.")
    print("=" * 70 + "\n")
    sys.exit(1)


def is_network_host_reachable(url: str, timeout: float = 2.0) -> bool:
    """
    Quick non-blocking probe to verify if the stream host:port is reachable.
    Avoids long (20-30s) connection hangs when ESP32-CAM is powered off.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        return True  # Local device index or local file

    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            return True

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.close()
        return True
    except (socket.timeout, socket.error, OSError):
        return False


def create_camera_capture(stream_url: str):
    """
    Initializes and validates video capture from ESP32-CAM or local webcam.
    Provides diagnostic error reporting on connection failure.
    """
    is_http = isinstance(stream_url, str) and (stream_url.startswith("http://") or stream_url.startswith("https://"))

    print(f"[CAM] Connecting to video source: {stream_url} ...")

    # Fast network check for HTTP streams
    if is_http:
        print("[CAM] Probing network reachability to ESP32-CAM...")
        if not is_network_host_reachable(stream_url, timeout=2.5):
            print("\n" + "=" * 70)
            print(f" [ERROR] COULD NOT REACH ESP32-CAM AT: {stream_url}")
            print("=" * 70)
            print(" Troubleshooting Checklist:")
            print("  1. Ensure the ESP32-CAM is powered ON (LED / power indicator active).")
            print("  2. Ensure your PC is connected to the ESP32-CAM Wi-Fi Access Point")
            print("     (e.g., 'RoboForensic_CAM' or local router IP).")
            print(f"  3. Open a browser and verify stream URL responds: {stream_url}")
            print("  4. To test with a local webcam instead, run with: --stream 0")
            print("=" * 70 + "\n")
            sys.exit(1)

    # Open VideoCapture
    # If device index passed as string (e.g. "0")
    if isinstance(stream_url, str) and stream_url.isdigit():
        source = int(stream_url)
    else:
        source = stream_url

    # For webcams on Windows, cv2.CAP_DSHOW prevents startup delays
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
        # Minimize internal buffer lag for network MJPEG streams
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    if not cap.isOpened():
        print("\n" + "=" * 70)
        print(f" [ERROR] FAILED TO OPEN VIDEO STREAM: {stream_url}")
        print("=" * 70)
        print(" OpenCV VideoCapture could not initialize the stream.")
        print(" Please verify that:")
        print("  - The camera is not open in another browser tab or application.")
        print("  - The stream URL is streaming valid MJPEG video.")
        print("=" * 70 + "\n")
        sys.exit(1)

    print("[CAM] Video stream connected successfully!")
    return cap


# ==============================================================================
# ZONE AND TARGET LOGIC
# ==============================================================================
def calculate_zones(frame_width: int):
    """
    Divides frame into 3 vertical zones:
    LEFT:   0           to W // 3
    CENTER: W // 3      to 2 * (W // 3)
    RIGHT:  2 * (W // 3) to W
    """
    left_boundary = frame_width // 3
    right_boundary = 2 * (frame_width // 3)
    return left_boundary, right_boundary


def get_position_zone(center_x: int, left_boundary: int, right_boundary: int) -> str:
    """
    Determines whether center_x falls in LEFT, CENTER, or RIGHT zone.
    """
    if center_x < left_boundary:
        return "LEFT"
    elif center_x > right_boundary:
        return "RIGHT"
    else:
        return "CENTER"


def select_primary_target(detections):
    """
    Selects the most relevant single target when multiple detections occur.
    Priority: Largest bounding box area with valid confidence.
    """
    if not detections:
        return None
    return max(detections, key=lambda d: d["area"])


# ==============================================================================
# VISUAL HUD RENDERING
# ==============================================================================
def draw_hud(frame, detections, primary_target, left_boundary, right_boundary, fps, conf_threshold):
    """
    Renders clean, informative HUD overlays on the live camera frame:
    - Top telemetry bar (FPS on left, stream status/mode, Confidence threshold on right)
    - 3-Zone vertical dividing lines & cleanly centered zone badges
    - All detected bounding boxes
    - Primary target highlighted with center point crosshairs and details
    - Bottom telemetry bar (Target info or 'NO RED OBJECT DETECTED')
    """
    h, w, _ = frame.shape
    active_zone = primary_target["zone"] if primary_target else None

    # 1. TOP TELEMETRY BAR
    header_h = 32
    header_overlay = frame.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, header_h), (18, 18, 18), -1)
    cv2.addWeighted(header_overlay, 0.60, frame, 0.40, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Mode info in center of top bar
    cv2.putText(frame, "ROBOFORENSIC AI - VISION ONLY", (w // 2 - 110, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"CONF: {conf_threshold:.2f}", (w - 95, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # 2. ZONE DIVIDING LINES & ZONE BADGES
    line_color = (90, 90, 90)
    cv2.line(frame, (left_boundary, header_h), (left_boundary, h - 55), line_color, 1, cv2.LINE_AA)
    cv2.line(frame, (right_boundary, header_h), (right_boundary, h - 55), line_color, 1, cv2.LINE_AA)

    # Zone badges placed below header at y=52
    badge_y = 54
    zones_def = [
        ("LEFT", left_boundary // 2, active_zone == "LEFT"),
        ("CENTER", (left_boundary + right_boundary) // 2, active_zone == "CENTER"),
        ("RIGHT", (right_boundary + w) // 2, active_zone == "RIGHT")
    ]

    for name, center_pos, is_active in zones_def:
        if is_active:
            pill_color = (0, 215, 255)  # Bright Amber / Gold
            text = f"[{name}]"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (center_pos - tw // 2 - 8, badge_y - th - 5),
                          (center_pos + tw // 2 + 8, badge_y + 5), pill_color, -1)
            cv2.putText(frame, text, (center_pos - tw // 2, badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            text = name
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            # Subtle dark background pill for readability over video backgrounds
            cv2.rectangle(frame, (center_pos - tw // 2 - 6, badge_y - th - 4),
                          (center_pos + tw // 2 + 6, badge_y + 4), (30, 30, 30), -1)
            cv2.putText(frame, text, (center_pos - tw // 2, badge_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

    # 3. DRAW ALL DETECTIONS
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        is_primary = (primary_target is not None and det is primary_target)

        if is_primary:
            # Highlighted Primary Target: Bright Green & Bold
            box_color = (0, 255, 0)
            thickness = 2
            cx, cy = det["center"]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # Center point marker
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            # Center crosshairs
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1, cv2.LINE_AA)

            # Label above box with background tag
            label = f"{det['class_name']}: {det['conf']*100:.1f}% | ({cx},{cy})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(header_h + 5, y1 - lh - 8)
            cv2.rectangle(frame, (x1, tag_y1), (x1 + lw + 8, tag_y1 + lh + 6), (0, 180, 0), -1)
            cv2.putText(frame, label, (x1 + 4, tag_y1 + lh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # Secondary Detections: Orange thinner box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 1)
            label = f"{det['class_name']}: {det['conf']*100:.1f}%"
            cv2.putText(frame, label, (x1, max(header_h + 5, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)

    # 4. BOTTOM TELEMETRY BANNER
    footer_h = 56
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, h - footer_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(footer_overlay, 0.75, frame, 0.25, 0, frame)

    if primary_target:
        cx, cy = primary_target["center"]
        zone_str = primary_target["zone"]
        conf_pct = primary_target["conf"] * 100.0

        # Target telemetry
        status_line1 = f"TARGET: {primary_target['class_name']}   CONF: {conf_pct:.1f}%   CENTER: ({cx}, {cy})"
        cv2.putText(frame, status_line1, (16, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # Position Badge
        zone_color = (0, 255, 0) if zone_str == "CENTER" else (0, 215, 255)
        pos_text = f"POSITION: {zone_str}"
        cv2.putText(frame, pos_text, (16, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, zone_color, 2, cv2.LINE_AA)
    else:
        # NO RED OBJECT DETECTED banner
        no_obj_text = "NO RED OBJECT DETECTED"
        cv2.putText(frame, no_obj_text, (16, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 70, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Scanning live camera feed for target...", (16, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Key hint at bottom-right
    cv2.putText(frame, "Press 'Q' or ESC to Exit", (w - 170, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


# ==============================================================================
# MAIN LIVE DETECTION ENGINE
# ==============================================================================
def run_live_detection(model_path: str = DEFAULT_MODEL_PATH,
                       stream_url: str = DEFAULT_STREAM_URL,
                       conf_threshold: float = DEFAULT_CONF_THRESHOLD):
    """
    Main live detection loop:
    1. Verifies YOLO model existence.
    2. Loads model into memory.
    3. Connects to ESP32-CAM stream.
    4. Continuously captures frames, runs inference, draws HUD, prints terminal telemetry.
    5. Cleanly handles frame drops and user shutdown keys.
    """
    print("\n" + "=" * 70)
    print(" ROBOFORENSIC ROBOT - LIVE RED OBJECT DETECTION MODULE")
    print("=" * 70)
    print(f" Target Model        : {model_path}")
    print(f" Camera Stream       : {stream_url}")
    print(f" Confidence Threshold: {conf_threshold:.2f}")
    print(" Control Mode        : Detection Only (Rover Motors / Arm INACTIVE)")
    print(" Exit Hotkeys        : [Q] or [ESC]")
    print("-" * 70)

    # STEP 1: Verify Model
    valid_model_path = verify_model_exists(model_path)
    print(f"[AI] Loading YOLO model from '{valid_model_path}'...")
    try:
        model = YOLO(valid_model_path)
        print("[AI] YOLO model loaded successfully.")
        print(f"[AI] Model classes: {model.names}")
    except Exception as e:
        print(f"[ERROR] Failed to instantiate YOLO model: {e}")
        sys.exit(1)

    # STEP 2: Connect Camera
    cap = create_camera_capture(stream_url)

    # Window creation
    window_name = "RoboForensic - Live Red Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\n[SYSTEM] Live detection pipeline running.")
    print("[SYSTEM] Video window active. Press 'Q' or 'ESC' on video window to quit.\n")

    # Metrics and loop state
    fps_timer = time.time()
    fps_frame_count = 0
    fps_display = 0.0

    last_terminal_log_time = 0.0
    terminal_log_interval = 0.35  # Log terminal every 350ms to keep output clear & responsive

    consecutive_frame_errors = 0
    max_frame_errors = 30  # Allow brief camera stalls without crashing

    try:
        while True:
            # 1. Grab Frame
            ret, frame = cap.read()

            # Handle Camera Frame Failure gracefully
            if not ret or frame is None:
                consecutive_frame_errors += 1
                if consecutive_frame_errors == 1:
                    print("[WARNING] Frame capture returned empty/corrupted frame. Retrying...")

                if consecutive_frame_errors > max_frame_errors:
                    print("\n[ERROR] Excessive frame read failures from camera stream.")
                    print("[ERROR] ESP32-CAM stream may have disconnected. Stopping pipeline.")
                    break

                # Render standby frame in window so user sees status
                standby = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(standby, "WAITING FOR ESP32-CAM STREAM...", (80, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(standby, f"Stream: {stream_url}", (80, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(standby, "Press 'Q' or ESC to Cancel", (80, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
                cv2.imshow(window_name, standby)

                key = cv2.waitKey(100) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("\n[SYSTEM] Shutdown requested by user during standby.")
                    break
                continue

            # Frame read succeeded: reset error counter
            consecutive_frame_errors = 0

            # Compute FPS
            fps_frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = fps_frame_count / (now - fps_timer)
                fps_frame_count = 0
                fps_timer = now

            # Dimensions and Zones
            frame_h, frame_w, _ = frame.shape
            left_boundary, right_boundary = calculate_zones(frame_w)

            # 2. Real-Time YOLO Inference
            # imgsz=416 or 320 for ultra-fast, smooth inference
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue

                    cls_id = int(box.cls[0])
                    class_name = model.names.get(cls_id, TARGET_CLASS_NAME)

                    # Extract coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Compute center
                    cx = int((x1 + x2) / 2.0)
                    cy = int((y1 + y2) / 2.0)

                    # Compute area for primary target selection
                    area = (x2 - x1) * (y2 - y1)

                    # Determine zone
                    zone = get_position_zone(cx, left_boundary, right_boundary)

                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "conf": conf,
                        "center": (cx, cy),
                        "area": area,
                        "zone": zone,
                        "class_name": class_name
                    })

            # 3. Select Single Primary Target (Largest Detection)
            primary_target = select_primary_target(detections)

            # 4. Terminal Output (Throttled for clear readability)
            if now - last_terminal_log_time >= terminal_log_interval:
                if primary_target:
                    cx, cy = primary_target["center"]
                    conf_pct = primary_target["conf"] * 100.0
                    zone = primary_target["zone"]
                    num_targets = len(detections)
                    multi_tag = f" (1 of {num_targets})" if num_targets > 1 else ""
                    print(f"[DETECTED] red_object{multi_tag} | Confidence: {conf_pct:.1f}% | Center: X={cx}, Y={cy} | POSITION: {zone}")
                else:
                    print("[STATUS] NO RED OBJECT DETECTED")
                last_terminal_log_time = now

            # 5. Render Video HUD Overlay
            draw_hud(
                frame=frame,
                detections=detections,
                primary_target=primary_target,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                fps=fps_display,
                conf_threshold=conf_threshold
            )

            # 6. Display Video Feed
            cv2.imshow(window_name, frame)

            # 7. Check for Exit Hotkeys (Q, q, or ESC)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("\n[SYSTEM] Exit key pressed ('Q' or ESC). Initiating clean shutdown...")
                break

    except KeyboardInterrupt:
        print("\n[SYSTEM] KeyboardInterrupt (Ctrl+C) received. Stopping...")

    finally:
        # STEP 5: Clean Release of Resources
        print("[SYSTEM] Releasing camera capture device...")
        if cap is not None:
            cap.release()
        print("[SYSTEM] Closing OpenCV display windows...")
        cv2.destroyAllWindows()
        print("[SYSTEM] Live detection pipeline closed cleanly.")


# ==============================================================================
# CLI PARSER
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="RoboForensic Robot - Live Detection Module (ESP32-CAM + YOLO red_object)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained YOLO model weights (.pt)"
    )
    parser.add_argument(
        "--stream",
        type=str,
        default=DEFAULT_STREAM_URL,
        help="ESP32-CAM MJPEG stream URL (or local webcam index like 0)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help="Detection confidence threshold"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_live_detection(
        model_path=args.model,
        stream_url=args.stream,
        conf_threshold=args.conf
    )
