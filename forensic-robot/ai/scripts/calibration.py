import cv2
import sys
import os
from distance import DistanceEstimator

def calibrate_camera(stream_url=0):
    """
    Interactive Distance Calibration script for ESP32-CAM / Webcam.
    
    Steps:
    1. Place your target red object at a measured distance (e.g., 50 cm).
    2. Input the actual object width (cm) and measured distance (cm).
    3. Click and drag or use YOLO bounding box to capture target width in pixels.
    4. Calculates and saves focal length to calibration.json.
    """
    print("==================================================")
    print(" ESP32-CAM Monocular Distance Calibration Utility ")
    print("==================================================")
    
    try:
        known_width = float(input("Enter actual object width in cm (e.g. 10): "))
        measured_distance = float(input("Enter measured distance from camera in cm (e.g. 50): "))
    except ValueError:
        print("Invalid input! Please enter valid numeric values.")
        return

    estimator = DistanceEstimator()
    
    # Open camera stream (URL string or integer index)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Unable to open camera stream at {stream_url}")
        return

    print("\nInstructions:")
    print(" - Press 'c' to capture current frame and enter pixel width manually.")
    print(" - Press 's' to select bounding box ROI with mouse.")
    print(" - Press 'q' to quit without saving.")

    bbox_pixel_width = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from stream.")
            break

        cv2.putText(frame, "Press 's' to select ROI bounding box | 'q' to Quit", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Camera Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Select ROI using OpenCV UI
            roi = cv2.selectROI("Select Object Bounding Box", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object Bounding Box")
            x, y, w, h = roi
            if w > 0:
                bbox_pixel_width = w
                print(f"\nCaptured object pixel width: {bbox_pixel_width} px")
                break
        elif key == ord('c'):
            try:
                bbox_pixel_width = float(input("Enter detected object pixel width: "))
                break
            except ValueError:
                print("Invalid number!")
        elif key == ord('q'):
            print("Calibration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if bbox_pixel_width > 0:
        focal_length = estimator.calculate_focal_length(measured_distance, bbox_pixel_width, known_width)
        print("\n--------------------------------------------------")
        print(f" Calculated Focal Length : {focal_length:.2f} pixels")
        print(f" Object Width           : {known_width} cm")
        print(f" Reference Distance     : {measured_distance} cm")
        print(f" Reference Pixel Width  : {bbox_pixel_width} px")
        print("--------------------------------------------------")
        
        save = input("Save calibration to calibration.json? (y/n): ").strip().lower()
        if save == 'y':
            estimator.save_calibration(focal_length, known_width)

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.4.1/stream"
    # If integer string like "0", parse as webcam index
    if url.isdigit():
        url = int(url)
    calibrate_camera(url)
