import os
import sys
import time
import json
import threading
import urllib.request
import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import serial.tools.list_ports

# Add scripts directory to module path
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from distance import DistanceEstimator
from rover_control import RoverController

app = FastAPI(title="RoboForensic High-Performance Zero-Lag Teleoperation Station")

web_dir = os.path.join(current_dir, "web")
os.makedirs(web_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=web_dir), name="static")

# Shared Application State
state_lock = threading.Lock()
app_state = {
    "stream_url": "http://192.168.4.1/stream",
    "serial_port": "COM21",
    "mode": "autonomous",  # "autonomous" or "manual"
    "conf_threshold": 0.35,
    "stop_distance": 25.0,
    "target_detected": False,
    "confidence": 0.0,
    "distance_cm": None,
    "zone": "NO TARGET",
    "command_code": "SEARCH",
    "command_desc": "Searching for target...",
    "fps": 0.0,
    "stream_status": "Active Stream",
    "serial_connected": False,
    "manual_cmd": "S",
    "calibration_file": os.path.join(scripts_dir, "calibration.json"),
    "logs": []
}

def log_event(msg):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    with state_lock:
        app_state["logs"].append(entry)
        if len(app_state["logs"]) > 50:
            app_state["logs"].pop(0)
    print(entry)

# Load YOLO Weights
model_path = os.path.join(current_dir, "models/best.pt")
if not os.path.exists(model_path):
    root_weights = os.path.join(current_dir, "../../yolo11n.pt")
    if os.path.exists(root_weights):
        model_path = root_weights
    else:
        model_path = "yolo11n.pt"

print(f"[Vision Engine] Loading YOLO weights from: {model_path}")
detector = YOLO(model_path)
# Warm up model at 320x320
dummy = np.zeros((320, 320, 3), dtype=np.uint8)
detector(dummy, imgsz=320, verbose=False)

distance_estimator = DistanceEstimator(config_path=app_state["calibration_file"])
rover_controller = None

def init_serial_controller(port_name):
    global rover_controller
    try:
        if rover_controller:
            rover_controller.close()
        rover_controller = RoverController(port=port_name, frame_width=640, stop_distance_cm=app_state["stop_distance"])
        is_conn = rover_controller.serial_conn is not None and rover_controller.serial_conn.is_open
        with state_lock:
            app_state["serial_connected"] = is_conn
            app_state["serial_port"] = port_name
        log_event(f"Serial controller on {port_name} (Connected: {is_conn})")
    except Exception as e:
        log_event(f"Serial error: {e}")
        with state_lock:
            app_state["serial_connected"] = False

init_serial_controller(app_state["serial_port"])

# ==============================================================================
# ZERO-LAG DECOUPLED FRAME GRABBER & INFERENCE PIPELINE
# ==============================================================================
latest_raw_frame = None
raw_frame_lock = threading.Lock()

latest_detection = {
    "box": None,
    "confidence": 0.0,
    "distance_cm": None,
    "zone": "NO TARGET",
    "auto_cmd": "SEARCH",
    "cmd_desc": "Searching...",
    "detected": False
}
detection_lock = threading.Lock()

stream_thread_running = True

def fast_stream_receiver():
    """
    Dedicated thread that continuously reads from the stream source.
    CRITICAL: Always keeps ONLY the freshest frame and immediately discards
    any backlog buffer, completely eliminating the 3-5 second OpenCV lag!
    """
    global latest_raw_frame, stream_thread_running
    current_source = None
    cap = None

    while stream_thread_running:
        with state_lock:
            source = app_state["stream_url"]

        is_webcam = str(source).isdigit()
        parsed_source = int(source) if is_webcam else source

        # If source is an HTTP MJPEG stream (ESP32-CAM), read directly from socket buffer
        if not is_webcam and str(source).startswith("http"):
            try:
                with state_lock:
                    app_state["stream_status"] = f"Connecting to {source}..."
                req = urllib.request.Request(source, headers={'User-Agent': 'RoboForensic/2.0'})
                stream = urllib.request.urlopen(req, timeout=4.0)
                bytes_buffer = b''
                with state_lock:
                    app_state["stream_status"] = "Active Stream"
                log_event(f"Zero-lag direct stream connected: {source}")

                while stream_thread_running:
                    # Check if source changed
                    with state_lock:
                        if app_state["stream_url"] != source:
                            break

                    chunk = stream.read(4096)
                    if not chunk:
                        time.sleep(0.01)
                        continue
                    bytes_buffer += chunk

                    # Look for JPEG boundaries: SOI (\xff\xd8) and EOI (\xff\xd9)
                    a = bytes_buffer.find(b'\xff\xd8')
                    if a != -1:
                        if a > 0:
                            bytes_buffer = bytes_buffer[a:]
                            a = 0
                        b = bytes_buffer.find(b'\xff\xd9', 2)
                        if b != -1:
                            jpg_data = bytes_buffer[:b+2]
                            bytes_buffer = bytes_buffer[b+2:]  # Flush read bytes immediately!

                        # Fast decode
                        frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = cv2.resize(frame, (640, 480))
                            with raw_frame_lock:
                                latest_raw_frame = frame
            except Exception as e:
                # Standby frame on error
                standby = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(standby, "STREAM RECONNECTING...", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(standby, f"{source}", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
                with raw_frame_lock:
                    latest_raw_frame = standby
                with state_lock:
                    app_state["stream_status"] = "Reconnecting..."
                time.sleep(1.0)
                continue

        else:
            # Local Webcam (Index 0, 1, etc.)
            if cap is None or not cap.isOpened() or current_source != parsed_source:
                if cap:
                    cap.release()
                current_source = parsed_source
                cap = cv2.VideoCapture(parsed_source, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                with state_lock:
                    app_state["stream_status"] = "Active Stream"
                log_event(f"Webcam {parsed_source} initialized (Zero-Lag buffer=1)")

            # Continuously grab to flush any driver buffer
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.resize(frame, (640, 480))
                with raw_frame_lock:
                    latest_raw_frame = frame
            else:
                time.sleep(0.02)

    if cap:
        cap.release()

# Start background stream grabber
receiver_thread = threading.Thread(target=fast_stream_receiver, daemon=True)
receiver_thread.start()

def async_inference_worker():
    """
    Dedicated AI Worker: Runs YOLO at imgsz=320 asynchronously (~24ms).
    Calculates distance and updates latest detection state without slowing
    down the video stream!
    """
    global stream_thread_running, latest_detection
    last_motor_time = 0

    while stream_thread_running:
        with raw_frame_lock:
            if latest_raw_frame is None:
                time.sleep(0.01)
                continue
            frame_to_process = latest_raw_frame.copy()

        with state_lock:
            conf_thresh = app_state["conf_threshold"]
            current_mode = app_state["mode"]
            stop_dist = app_state["stop_distance"]

        # Run fast YOLO inference at 320x320 on CPU (~23-25ms!)
        results = detector(frame_to_process, imgsz=320, conf=conf_thresh, verbose=False)[0]

        best_box = None
        max_area = 0
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (float(x1), float(y1), float(x2), float(y2), float(box.conf[0]))

        target_detected = False
        dist_cm = None
        auto_cmd = "SEARCH"
        zone_label = "NO TARGET"
        cmd_desc = "Searching field of view..."
        target_conf = 0.0

        if best_box:
            target_detected = True
            x1, y1, x2, y2, conf = best_box
            target_conf = conf
            box_width = x2 - x1
            target_cx = (x1 + x2) / 2.0

            dist_cm = distance_estimator.estimate_distance(box_width, smooth=True)
            if rover_controller:
                rover_controller.stop_distance_cm = stop_dist
                auto_cmd, zone_label, cmd_desc = rover_controller.determine_action(target_cx, dist_cm)
        else:
            distance_estimator.reset_history()

        # Update latest detection cache
        with detection_lock:
            latest_detection["box"] = best_box
            latest_detection["confidence"] = round(target_conf, 2)
            latest_detection["distance_cm"] = round(dist_cm, 1) if dist_cm is not None else None
            latest_detection["zone"] = zone_label
            latest_detection["auto_cmd"] = auto_cmd
            latest_detection["cmd_desc"] = cmd_desc
            latest_detection["detected"] = target_detected

        # Update telemetry state
        with state_lock:
            app_state["target_detected"] = target_detected
            app_state["confidence"] = round(target_conf, 2)
            app_state["distance_cm"] = round(dist_cm, 1) if dist_cm is not None else None
            app_state["zone"] = zone_label
            if current_mode == "autonomous":
                app_state["command_code"] = auto_cmd
                app_state["command_desc"] = cmd_desc
            else:
                app_state["command_code"] = app_state["manual_cmd"]
                app_state["command_desc"] = f"Manual Pilot Executing: {app_state['manual_cmd']}"

        # Send motor command periodically (every 180ms)
        now = time.time()
        if now - last_motor_time >= 0.18:
            last_motor_time = now
            if rover_controller:
                cmd_to_send = app_state["command_code"]
                rover_controller.send_command(cmd_to_send)

        # Brief pause to prevent CPU burn
        time.sleep(0.015)

# Start background AI worker
inference_thread = threading.Thread(target=async_inference_worker, daemon=True)
inference_thread.start()

# ==============================================================================
# FAST STREAM SERVING & HUD COMPOSITOR
# ==============================================================================
@app.get("/")
def index_page():
    return FileResponse(os.path.join(web_dir, "index.html"))

@app.get("/video_feed")
def video_feed():
    def frame_generator():
        fps_counter = 0
        fps_timer = time.time()

        while True:
            t0 = time.time()

            with raw_frame_lock:
                if latest_raw_frame is None:
                    time.sleep(0.02)
                    continue
                display_frame = latest_raw_frame.copy()

            h, w, _ = display_frame.shape

            # Overlay latest AI detection (Zero-Lag smooth overlay)
            with detection_lock:
                det = dict(latest_detection)

            if det["detected"] and det["box"]:
                x1, y1, x2, y2, conf = det["box"]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                dist_cm = det["distance_cm"]

                # Draw high-tech bounding box
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 230, 255), 2)
                # Corner accents
                c_len = 12
                for px, py in [(int(x1), int(y1)), (int(x2), int(y1)), (int(x1), int(y2)), (int(x2), int(y2))]:
                    cv2.circle(display_frame, (px, py), 3, (0, 255, 100), -1)
                
                # Crosshair
                cv2.drawMarker(display_frame, (int(cx), int(cy)), (0, 255, 120), cv2.MARKER_CROSS, 14, 2)
                lbl = f"TARGET: {conf*100:.0f}% | {dist_cm:.1f}cm" if dist_cm else f"TARGET: {conf*100:.0f}%"
                cv2.putText(display_frame, lbl, (int(x1), max(22, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Zone division lines
            z_left = int(w / 3)
            z_right = int(2 * w / 3)
            cv2.line(display_frame, (z_left, 0), (z_left, h), (60, 70, 90), 1, cv2.LINE_AA)
            cv2.line(display_frame, (z_right, 0), (z_right, h), (60, 70, 90), 1, cv2.LINE_AA)

            # Minimal Top HUD Strip
            hud = display_frame.copy()
            cv2.rectangle(hud, (0, 0), (w, 32), (10, 14, 22), -1)
            cv2.addWeighted(hud, 0.85, display_frame, 0.15, 0, display_frame)
            with state_lock:
                mode_str = app_state["mode"].upper()
                cmd_str = app_state["command_code"]
            cv2.putText(display_frame, f"MODE: {mode_str}", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display_frame, f"ZONE: {det['zone']}", (180, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cmd_col = (0, 255, 120) if cmd_str == 'F' else (0, 80, 255) if cmd_str == 'S' else (0, 220, 255)
            cv2.putText(display_frame, f"CMD: {cmd_str}", (360, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cmd_col, 2)
            d_str = f"{det['distance_cm']:.1f} cm" if det['distance_cm'] else "-- cm"
            cv2.putText(display_frame, f"DIST: {d_str}", (490, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1)

            # Fast JPEG compression (quality=65 for instant transfer and zero browser lag)
            ret, buffer = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            # Calculate actual FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                with state_lock:
                    app_state["fps"] = round(fps_counter / (time.time() - fps_timer), 1)
                fps_counter = 0
                fps_timer = time.time()

            # Target ~35-40 FPS with zero lag
            elapsed = time.time() - t0
            sleep_time = max(0.001, 0.025 - elapsed)
            time.sleep(sleep_time)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

# ==============================================================================
# REST API ENDPOINTS
# ==============================================================================
@app.get("/api/telemetry")
def get_telemetry():
    with state_lock:
        data = dict(app_state)
    data.pop("calibration_file", None)
    return JSONResponse(data)

@app.get("/api/ports")
def get_serial_ports():
    ports = serial.tools.list_ports.comports()
    results = [{"port": p.device, "description": p.description} for p in ports]
    return JSONResponse(results)

@app.post("/api/command")
async def send_manual_command(request: Request):
    data = await request.json()
    cmd = data.get("command", "S").upper()
    if cmd in ['F', 'B', 'L', 'R', 'S']:
        with state_lock:
            app_state["manual_cmd"] = cmd
            if app_state["mode"] == "manual":
                app_state["command_code"] = cmd
                app_state["command_desc"] = f"Manual Pilot Executing: {cmd}"
        if rover_controller:
            rover_controller.send_command(cmd)
        log_event(f"Manual Command: {cmd}")
        return JSONResponse({"status": "ok", "command": cmd})
    return JSONResponse({"status": "error", "message": "Invalid command"}, status_code=400)

@app.post("/api/mode")
async def set_mode(request: Request):
    data = await request.json()
    new_mode = data.get("mode", "autonomous").lower()
    if new_mode in ["autonomous", "manual"]:
        with state_lock:
            app_state["mode"] = new_mode
            if new_mode == "manual":
                app_state["manual_cmd"] = "S"
                app_state["command_code"] = "S"
        if rover_controller:
            rover_controller.send_command("S")
        log_event(f"Mode switched to: {new_mode.upper()}")
        return JSONResponse({"status": "ok", "mode": new_mode})
    return JSONResponse({"status": "error", "message": "Invalid mode"}, status_code=400)

@app.post("/api/config")
async def update_config(request: Request):
    data = await request.json()
    with state_lock:
        if "stream_url" in data:
            app_state["stream_url"] = str(data["stream_url"]).strip()
        if "conf_threshold" in data:
            app_state["conf_threshold"] = float(data["conf_threshold"])
        if "stop_distance" in data:
            app_state["stop_distance"] = float(data["stop_distance"])
        serial_port = data.get("serial_port")

    if serial_port:
        init_serial_controller(serial_port)

    log_event(f"Config updated: Stream={app_state['stream_url']}, Conf={app_state['conf_threshold']}")
    return JSONResponse({"status": "ok", "config": app_state})

@app.get("/api/calibration")
def get_calibration():
    data = {
        "focal_length": distance_estimator.focal_length,
        "known_width": distance_estimator.known_width
    }
    return JSONResponse(data)

@app.post("/api/calibration")
async def save_calibration(request: Request):
    data = await request.json()
    focal_length = float(data.get("focal_length", 500.0))
    known_width = float(data.get("known_width", 10.0))
    distance_estimator.save_calibration(focal_length, known_width)
    log_event(f"Camera Calibrated: F={focal_length}px, Object Width={known_width}cm")
    return JSONResponse({"status": "ok", "focal_length": focal_length, "known_width": known_width})

if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = 8000
    print(f"=============================================================")
    print(f" RoboForensic Zero-Lag Engine Deployed!                      ")
    print(f" Local Web:   http://127.0.0.1:{port}                       ")
    print(f" Network Web: http://192.168.4.2:{port}                      ")
    print(f"=============================================================")
    uvicorn.run(app, host=host, port=port, log_level="warning")
