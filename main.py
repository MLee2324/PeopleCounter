import cv2
from ultralytics import YOLO

from drawers import (
    draw_person_box,
    draw_counter
)

# Instance variables
entered = 0
exited = 0
#line_x = 340

def detect_people(frame, model):
    global entered, exited
    results = model.track(frame, persist=True, conf=0.6)

    for box in results[0].boxes:
        if int(box.cls) == 0 and box.conf > 0.6:
            bbox = box.xyxy[0].cpu().numpy()
            if box.id is not None:
                track_id = int(box.id.item())
                person_entered, person_exited = draw_person_box(frame, bbox, int(box.cls), track_id)
                if person_entered:
                    entered += 1
                if person_exited:
                    exited += 1
    return entered, exited

def main():
    model = YOLO("yolov8n.pt")

    # Load the video from 'videos' folder
    cap = cv2.VideoCapture("videos/handlep.mp4")

    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        

        detect_people(frame, model)
        draw_counter(frame, f"Exited: {exited}", (10, 40), (0, 0, 255))
        draw_counter(frame, f"Entered: {entered}", (520, 40), (0, 255, 0))

        #cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 255, 255), 2)

        # Show frame
        cv2.imshow("Security System Checker", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
