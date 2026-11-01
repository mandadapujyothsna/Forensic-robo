import cv2
from ultralytics import YOLO

STREAM_URL = "http://192.168.4.1/stream"
MODEL_PATH = r".\models\best.pt"

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded!")
print("Connecting to ESP32-CAM...")

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("ERROR: Cannot open ESP32-CAM stream")
    print("Check http://192.168.4.1/stream")
    exit()

print("ESP32-CAM connected!")
print("Starting red-object detection...")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        continue

    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]

            print(
                f"Detected: {name} | "
                f"Confidence: {conf * 100:.1f}%"
            )

    cv2.imshow(
        "RoboForensic - Red Object Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()