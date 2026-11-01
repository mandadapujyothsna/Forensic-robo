import os
import sys
from ultralytics import YOLO

def train_red_object_detector(data_yaml="../dataset/red_object/data.yaml", epochs=50, imgsz=640, model_type="yolo11n.pt"):
    """
    Trains YOLO on the red_object dataset and saves best weights to models/best.pt.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.abspath(os.path.join(script_dir, data_yaml))
    models_dir = os.path.abspath(os.path.join(script_dir, "../models"))
    os.makedirs(models_dir, exist_ok=True)

    print("==================================================")
    print(" RoboForensic YOLO Red Object Training Workflow ")
    print("==================================================")
    print(f" Data Config : {yaml_path}")
    print(f" Base Model  : {model_type}")
    print(f" Epochs      : {epochs}")
    print(f" Image Size  : {imgsz}")
    print("--------------------------------------------------")

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found! Ensure dataset is properly configured.")
        return

    # Load base model (YOLOv8n / YOLOv11n)
    model = YOLO(model_type)

    # Start training
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        project=os.path.join(script_dir, "../runs"),
        name="red_object_run",
        exist_ok=True
    )

    # Locate trained best.pt
    best_weights = os.path.join(script_dir, "../runs/red_object_run/weights/best.pt")
    target_weights = os.path.join(models_dir, "best.pt")

    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, target_weights)
        print("--------------------------------------------------")
        print(f" Training Complete! Best weights copied to:\n {target_weights}")
        print("--------------------------------------------------")
    else:
        print(f"Warning: Could not find {best_weights}")

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    train_red_object_detector(epochs=epochs)
