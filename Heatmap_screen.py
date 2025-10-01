import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")  # Or your custom model path

# Start video capture (0=default webcam; change to your camera index or RTSP stream)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Read one frame to get dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read from video source.")
    cap.release()
    exit()

height, width, _ = frame.shape

# Initialize heatmap with zeros
heatmap = np.zeros((height, width), dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Frame capture failed; skipping...")
        continue

    # Run YOLO detection
    results = model(frame, conf=0.3)  # Lower conf if needed for more detections

    # Create a mask for current detections
    mask = np.zeros((height, width), dtype=np.float32)

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # Draw a bright spot in the mask at detection center
        cv2.circle(mask, (center_x, center_y), radius=15, color=1, thickness=-1)

    # Accumulate the mask into the heatmap
    heatmap += mask

    # Decay heatmap slowly to avoid infinite accumulation (optional but recommended)
    heatmap *= 0.95

    # Normalize heatmap to 0-255
    norm_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    norm_heatmap = np.uint8(norm_heatmap)

    # Apply colormap for visualization
    colored_heatmap = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original frame
    overlay = cv2.addWeighted(frame, 0.7, colored_heatmap, 0.5, 0)

    # Display the combined frame
    cv2.imshow('YOLOv8 Heatmap', overlay)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
