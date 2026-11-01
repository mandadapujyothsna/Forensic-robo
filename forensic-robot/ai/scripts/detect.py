import os
import sys
import cv2
from ultralytics import YOLO

def test_image_detection(image_path, model_path="../models/best.pt"):
    """
    Tests YOLO detection on a single image file, printing bounding box, center, and confidence.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_abs_path = os.path.abspath(os.path.join(script_dir, model_path))

    # Fallback to pretrained base model if custom model isn't trained yet
    if not os.path.exists(model_abs_path):
        print(f"Warning: Custom weights '{model_abs_path}' not found. Using default yolo11n.pt fallback...")
        model_abs_path = "yolo11n.pt"

    print(f"Loading YOLO Model: {model_abs_path}")
    model = YOLO(model_abs_path)

    image_abs_path = os.path.abspath(image_path)
    if not os.path.exists(image_abs_path):
        print(f"Error: Image file '{image_abs_path}' does not exist.")
        return

    frame = cv2.imread(image_abs_path)
    if frame is None:
        print(f"Error loading image '{image_abs_path}'")
        return

    results = model(frame, conf=0.4)[0]
    
    print("\n--- Detection Results ---")
    detected = False
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        print(f"Class: {class_name} | Confidence: {conf*100:.1f}%")
        print(f" Bounding Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
        print(f" Width: {int(width)} px, Height: {int(height)} px")
        print(f" Center: ({center_x:.1f}, {center_y:.1f})")
        detected = True

    if not detected:
        print("No objects detected above confidence threshold.")

    # Annotate image and show/save output
    annotated = results.plot()
    out_path = os.path.splitext(image_abs_path)[0] + "_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Annotated result saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <path_to_test_image.jpg>")
    else:
        test_image_detection(sys.argv[1])
