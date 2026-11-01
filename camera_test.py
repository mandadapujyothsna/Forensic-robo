import cv2

CAMERA_URL = "http://192.168.4.1/stream"

print("[CAM] Connecting to ESP32-CAM...")
print("[CAM] Stream:", CAMERA_URL)

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("[ERROR] Could not connect to ESP32-CAM")
    print("[ERROR] Check the ESP32-CAM Wi-Fi and stream URL.")
    exit()

print("[CAM] Connected successfully!")
print("[CAM] Press Q or ESC to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to read frame")
        continue

    cv2.imshow("ESP32-CAM Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("[CAM] Camera test stopped.")