import cv2
import os

STREAM_URL = "http://192.168.4.1/stream"

SAVE_DIR = "dataset/images/capture"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("❌ Could not connect to ESP32-CAM")
    exit()

print("✅ ESP32-CAM connected")
print("SPACE = Capture image")
print("Q = Quit")

count = len(os.listdir(SAVE_DIR)) + 1

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read camera frame")
        break

    cv2.imshow("ESP32-CAM Image Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = os.path.join(
            SAVE_DIR,
            f"red_object_{count:03d}.jpg"
        )

        cv2.imwrite(filename, frame)
        print(f"✅ Saved: {filename}")

        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Capture finished")