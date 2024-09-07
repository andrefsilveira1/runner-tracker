from ultralytics import YOLO
import cv2


# Setup
model = YOLO('yolov8n.pt')
video_path = 'marathon.mp4'  
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

    results = model(frame, classes=[0], conf=0.4)

    annotated_frame = results[0].plot() 

    cv2.imshow("Marathon Runners Detection", annotated_frame)

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
