import json
import os
from collections import deque

class DistanceEstimator:
    """
    Monocular Distance Estimator using pinhole camera model:
    D = (FocalLength * ActualWidth) / PixelWidth
    """
    def __init__(self, config_path="calibration.json", window_size=5):
        self.config_path = config_path
        self.focal_length = 500.0  # Default fallback focal length in pixels
        self.known_width = 10.0     # Default actual object width in cm
        self.history = deque(maxlen=window_size)
        self.load_calibration()

    def load_calibration(self):
        """Loads calibration constants from JSON file if available."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    self.focal_length = float(data.get("focal_length", self.focal_length))
                    self.known_width = float(data.get("known_width", self.known_width))
                    print(f"[DistanceEstimator] Loaded calibration: F={self.focal_length:.2f}px, W={self.known_width:.2f}cm")
            except Exception as e:
                print(f"[DistanceEstimator] Warning loading calibration: {e}")
        else:
            print(f"[DistanceEstimator] Config {self.config_path} not found. Using default F={self.focal_length}px")

    def save_calibration(self, focal_length, known_width):
        """Saves calibrated focal length and object width to JSON file."""
        self.focal_length = focal_length
        self.known_width = known_width
        data = {
            "focal_length": self.focal_length,
            "known_width": self.known_width
        }
        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[DistanceEstimator] Saved calibration to {self.config_path}")

    def calculate_focal_length(self, measured_distance_cm, pixel_width, known_width_cm):
        """
        Calculates focal length F = (PixelWidth * MeasuredDistance) / ActualWidth
        """
        if known_width_cm <= 0 or pixel_width <= 0:
            return 0.0
        return (pixel_width * measured_distance_cm) / known_width_cm

    def estimate_distance(self, pixel_width, smooth=True):
        """
        Estimates distance D in centimeters given bounding box pixel width P.
        Applies moving average smoothing if smooth=True.
        """
        if pixel_width <= 0:
            return None
        
        raw_distance = (self.focal_length * self.known_width) / float(pixel_width)
        
        if not smooth:
            return raw_distance

        self.history.append(raw_distance)
        smoothed_distance = sum(self.history) / len(self.history)
        return smoothed_distance

    def reset_history(self):
        """Clears moving average history buffer."""
        self.history.clear()
