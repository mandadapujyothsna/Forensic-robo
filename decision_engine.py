import serial
import time

class RobotController:
    def __init__(self, port="COM3", baudrate=9600):
        """
        Initializes the serial connection to the hardware (Arduino / ESP32).
        Change 'COM3' to match your system's serial port (e.g., '/dev/ttyUSB0' on Linux/Pi).
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Give the microcontroller time to reset after opening serial
            print(f"[+] Connected to Robot Hardware on {port}")
        except Exception as e:
            self.ser = None
            print(f"[-] Serial connection failed: {e}. Running in simulation mode.")

    def send_command(self, action_code):
        """
        Sends single-character commands to the microcontroller:
        'S' = Stop
        'F' = Fast Forward
        'L' = Slow Forward
        'R' = Rotate/Search
        """
        if self.ser and self.ser.is_open:
            self.ser.write(action_code.encode())

    def get_action_from_distance(self, distance_cm):
        """
        Evaluates distance to target and sends the corresponding hardware command.
        """
        if distance_cm == 0:
            self.send_command('R')
            return "SEARCHING", "Rotate to scan"
        elif distance_cm < 15.0:
            self.send_command('S')
            return "STOP", "Target reached / Take sample"
        elif distance_cm < 40.0:
            self.send_command('L')
            return "APPROACH_SLOW", "Slow forward approach"
        else:
            self.send_command('F')
            return "APPROACH_FAST", "Full speed forward"