# motion-detection-camera
# A Python + OpenCV-based real-time motion detector that triggers sound alerts, captures frames with timestamp, and logs events.

#Features
- Real-time motion detection using webcam
- Beep alert on detection
- Saves image and logs timestamp
- Future plans: email/SMS alerts, GUI integration

import cv2
import winsound
from datetime import datetime
import os

save_path = "captures"
os.makedirs(save_path, exist_ok=True)

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret1, frame1 = cam.read()
    ret2, frame2 = cam.read()

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 6000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        winsound.Beep(800, 400)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open("motion_log.txt", "a") as log:
            log.write(f"Motion detected at {timestamp}\n")

        filename_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f"{save_path}/motion_{filename_time}.jpg", frame1)

    if cv2.waitKey(10) == ord('q'):
        break

    cv2.imshow('Disha Cam', frame1)
