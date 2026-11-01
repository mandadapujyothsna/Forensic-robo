import cv2
 from ultralytics import YOLO

 # Load custom YOLO model trained on red_object
 model = YOLO("cam/forensic-robot/ai/models/best.pt")

 # ESP32-CAM MJPEG Stream URL
 stream_url = "http://192.168.4.1/stream" # Adjust to your IP/port
 cap = cv2.VideoCapture(stream_url)

 while cap.isOpened():
     ret, frame = cap.read()
     if not ret:
         break

     # Run YOLO inference
     results = model(frame)

     for r in results:
         boxes = r.boxes
         for box in boxes:
             # Get bounding box coordinates [x1, y1, x2, y2]
             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
             pixel_width = x2 - x1
             pixel_height = y2 - y1

             # Pass bounding box dimensions to distance module
             # distance = calculate_distance(pixel_width)