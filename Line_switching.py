
from ultralytics import YOLO
import cv2
import os


VIDEO_PATH = r"C:\Users\mithi\Downloads\dataset\training_videos\breaking video3.mp4"
SCALE = 0.7
MIN_CHANGE_FRAMES = 8      
ALERT_DISPLAY_FRAMES = 20
CONF_THRESH = 0.4

def open_capture(video_path: str):
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        print("Video not found. Using webcam 0.")
        cap = cv2.VideoCapture(0)
    return cap

def main():
    model = YOLO("yolov8n.pt")

    cap = open_capture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Unable to open video source.")
        return

    prev_lane = None
    change_counter = 0
    alert_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or cannot be read.")
            break

        if SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

        h, w = frame.shape[:2]
        lane_split_x = w // 2 

        results = model(frame, imgsz=640, verbose=False, conf=CONF_THRESH)

        best_score = -1
        best_box = None
        best_label = None

        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                if label not in ["car", "truck", "bus", "motorbike"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                score = area * (y2 / h)

                if score > best_score:
                    best_score = score
                    best_box = (x1, y1, x2, y2)
                    best_label = label

        if best_box:
            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) // 2
            current_lane = 1 if cx < lane_split_x else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 255, 60), 2)
            cv2.line(frame, (cx, y1), (cx, y2), (60, 255, 60), 1)
            cv2.putText(frame, f"{best_label} (lane {current_lane})", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

            if prev_lane is not None and current_lane != prev_lane:
                change_counter += 1
            else:
                change_counter = 0

            if change_counter >= MIN_CHANGE_FRAMES:
                alert_timer = ALERT_DISPLAY_FRAMES
                prev_lane = current_lane
                change_counter = 0
            elif prev_lane is None:
                prev_lane = current_lane

        cv2.line(frame, (lane_split_x, 0), (lane_split_x, h), (255, 160, 0), 2)
        cv2.putText(frame, "Lane 1", (lane_split_x // 2 - 40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, "Lane 2", (lane_split_x + w // 4 - 40, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        if alert_timer > 0:
            cv2.putText(frame, "LANE CHANGE DETECTED!", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            alert_timer -= 1

        cv2.imshow("Two-Lane Change Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
