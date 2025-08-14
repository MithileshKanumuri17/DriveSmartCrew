import cv2
import numpy as np
from collections import deque

# === Video path ===
video_path = r"C:\Users\mithi\Downloads\dataset\training_videos\breaking video2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
print("FPS:", fps)

# Calibration
pixels_per_meter = 60

# First frame
ret, prev_frame = cap.read()
if not ret or prev_frame is None:
    print("Error: Could not read the first frame.")
    cap.release()
    exit()

prev_frame = cv2.resize(prev_frame, (500, 500))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Good features to track
points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

# Variables
speed_history = deque(maxlen=10)  # store last 10 speed readings
current_speed = 0
previous_speed = 0
last_brake_detected = -999
frame_index = 0
sudden_brake_count = 0
alert_duration = 8
show_alert = False
alert_counter = 0
min_movement_threshold = 0.1  # pixels
min_speed_for_brake = 0.2 # km/h

# Main loop
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("End of video or failed to grab frame.")
        break

    frame_index += 1
    frame = cv2.resize(frame, (500, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if points is not None:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
        if new_points is not None:
            movement = np.linalg.norm(new_points - points, axis=2)
            avg_pixels_per_frame = np.median(movement)  # median is more robust

            # Ignore tiny movements (noise)
            if avg_pixels_per_frame < min_movement_threshold:
                avg_pixels_per_frame = 0

            meters_per_frame = avg_pixels_per_frame / pixels_per_meter
            meters_per_second = meters_per_frame * fps
            current_speed = meters_per_second * 3.6  # km/h

            # Store in history
            speed_history.append(current_speed)
            smoothed_speed = np.mean(speed_history)

            # Brake detection
            if (
                smoothed_speed < previous_speed * 0.6 and  # big drop
                previous_speed > min_speed_for_brake and
                frame_index - last_brake_detected > alert_duration
            ):
                sudden_brake_count += 1
                show_alert = True
                alert_counter = alert_duration
                last_brake_detected = frame_index

            points = new_points

    # Recalculate points every 50 frames to avoid drift
    if frame_index % 50 == 0:
        points = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

    # Alert control
    if alert_counter > 0:
        alert_counter -= 1
    else:
        show_alert = False

    # Display
    if show_alert:
        if not(sudden_brake_count == 1):
         cv2.putText(frame, "BRAKE ALERT!", (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Speed: {current_speed:.2f} km/h", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Motion Detection", frame)

    previous_speed = np.mean(speed_history) if speed_history else current_speed
    prev_gray = gray.copy()

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Total Sudden Brakes Detected: {sudden_brake_count}")
