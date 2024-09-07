from ultralytics import YOLO
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# Setup YOLO model
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Open video file
video_path = 'marathon.mp4'  
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Something went wrong")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('marathon_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Define the pixel-to-meter approximation (adjust as needed)
pixel_to_meter = 0.01  # Needs calibration for actual distances

previous_positions = {}

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get YOLO detections
    results = model(frame, classes=[0], conf=0.4)  # Detect persons (class 0)
    detections = []
    
    # Process each detection and extract bounding boxes and confidence scores
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box
        conf = float(result.conf)  # Confidence score (convert to float)
        detections.append([x1, y1, x2, y2, conf])  # Ensure correct format: [x1, y1, x2, y2, confidence]

    # Ensure detections list has the correct format before passing to tracker
    if len(detections) > 0:
        tracked_objects = tracker.update_tracks(detections, frame)  # Update tracker

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj[:5])

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Speed calculation (if we have previous position for this track ID)
            if track_id in previous_positions:
                prev_x, prev_y = previous_positions[track_id]
                current_x, current_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Calculate distance in pixels
                pixel_distance = np.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)
                real_distance = pixel_distance * pixel_to_meter  # Convert to real distance
                speed = real_distance * fps  # Speed in meters per second

                cv2.putText(frame, f'Speed: {speed:.2f} m/s', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            # Update previous positions
            previous_positions[track_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Show the frame and write to output
    cv2.imshow('Tracking Athletes', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
