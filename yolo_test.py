from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')

video_path = 'marathon.mp4'  # Replace with the path to your marathon video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Something goes wrong")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('marathon_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Use YOLO model for object detection on the current frame
    results = model(frame, classes=[0], conf=0.4)

    # Display and annotate results
    annotated_frame = results[0].plot()  # Annotates the detected objects on the frame

    # Show the frame
    cv2.imshow("Marathon Runners Detection", annotated_frame)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
