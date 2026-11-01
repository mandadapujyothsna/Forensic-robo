import time

try:
    import serial
except ImportError:
    serial = None

class RoverController:
    """
    Rover Navigation & Decision Engine:
    - Partition frame into LEFT, CENTER, RIGHT zones.
    - Evaluate distance relative to safe stopping threshold.
    - Emit motor commands ('F', 'B', 'L', 'R', 'S') via Serial / BT.
    """
    def __init__(self, port=None, baudrate=115200, frame_width=640, stop_distance_cm=25.0):
        self.frame_width = frame_width
        self.stop_distance_cm = stop_distance_cm
        self.port = port
        self.serial_conn = None
        
        # Define 3 Zone Boundaries (0 .. L_ZONE .. R_ZONE .. frame_width)
        self.left_threshold = int(frame_width / 3)        # e.g., 213 px for 640 width
        self.right_threshold = int(2 * frame_width / 3)   # e.g., 426 px for 640 width

        if self.port:
            self.connect_serial(port, baudrate)

    def connect_serial(self, port, baudrate):
        """Initializes serial connection to ESP32 motor controller."""
        if serial is None:
            print("[RoverController] 'pyserial' package is not installed. Running in simulation mode.")
            return

        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for serial initialization
            print(f"[RoverController] Connected to serial port {port} at {baudrate} baud.")
        except Exception as e:
            print(f"[RoverController] Warning: Could not open serial port {port}: {e}")
            self.serial_conn = None

    def determine_action(self, target_center_x, distance_cm):
        """
        Determines the rover motion command based on target position and distance.
        Returns tuple: (Command_Code, Zone_Label, Description)
        Command Codes: 'L', 'R', 'F', 'S', 'SEARCH'
        """
        if target_center_x is None or distance_cm is None:
            return ("SEARCH", "NO TARGET", "Keep Searching - Rotating / Idle")

        # Determine Zone
        if target_center_x < self.left_threshold:
            zone = "LEFT"
        elif target_center_x > self.right_threshold:
            zone = "RIGHT"
        else:
            zone = "CENTER"

        # Evaluate Distance & Navigation State
        if distance_cm <= self.stop_distance_cm:
            return ("S", zone, f"Target Reached ({distance_cm:.1f} cm <= {self.stop_distance_cm} cm) -> STOP")
        
        if zone == "LEFT":
            return ("L", zone, f"Target in LEFT zone ({target_center_x:.0f}px) -> TURN LEFT")
        elif zone == "RIGHT":
            return ("R", zone, f"Target in RIGHT zone ({target_center_x:.0f}px) -> TURN RIGHT")
        else: # CENTER
            return ("F", zone, f"Target CENTERED ({target_center_x:.0f}px, {distance_cm:.1f} cm) -> FORWARD")

    def send_command(self, command_code):
        """Transmits single-character motor command to ESP32 over serial if connected."""
        if command_code == "SEARCH":
            # For search mode, we don't spam commands or we can send periodic spin/stop
            return

        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(command_code.encode('utf-8'))
                print(f"[RoverController] Sent command: {command_code}")
            except Exception as e:
                print(f"[RoverController] Error sending command '{command_code}': {e}")
        else:
            # Console simulation mode
            pass

    def close(self):
        """Closes serial connection safely."""
        if self.serial_conn and self.serial_conn.is_open:
            self.send_command("S") # Send stop on shutdown
            self.serial_conn.close()
            print("[RoverController] Serial port closed.")
