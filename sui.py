import cv2

import serial

import time

from util import get_limits

COM_PORT = 'COM6'  
BAUD_RATE = 9600

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print(f"Connected to {COM_PORT}")
except Exception as e:
    print(f"Could not open serial port: {e}")
    ser = None


yellow = [0, 255, 255]
lime_green = [50, 205, 50]

MIN_AREA = 100  # minimum contour area to count as a detection (tune as needed)




IP_WEBCAM_URL = "https://10.160.76.113:8080/video"

cap = cv2.VideoCapture("https://10.101.12.212:8080/video")
#cap = cv2.VideoCapture(0)

last_sent = None  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Yellow mask
    yLower, yUpper = get_limits(color=yellow)
    yellow_mask = cv2.inRange(hsvImage, yLower, yUpper)

    # Lime green mask
    gLower, gUpper = get_limits(color=lime_green)
    lime_green_mask = cv2.inRange(hsvImage, gLower, gUpper)

    # Find largest contour area for each mask to decide which color is "detected"
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(lime_green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yellow_area = max((cv2.contourArea(c) for c in yellow_contours), default=0)
    green_area = max((cv2.contourArea(c) for c in green_contours), default=0)

    detected = None
    if yellow_area > MIN_AREA or green_area > MIN_AREA:
        if yellow_area > green_area:
            detected = 'Y'
        else:
            detected = 'R'

    # Send over serial only when detection changes (debounced)
    # Only send if we have an actual detection AND it's different from last sent
    # This prevents spurious color changes due to momentary loss of detection
    if detected is not None and detected != last_sent:
        if ser is not None:
            ser.write(detected.encode('utf-8'))
            print(f"Sent: {detected}")
        last_sent = detected

    # ---- Visualization ----
    if detected == 'Y' and yellow_area > MIN_AREA:
        yellow_box = cv2.boundingRect(yellow_mask)
        x, y, w, h = yellow_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "Yellow", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif detected == 'R' and green_area > MIN_AREA:
        green_box = cv2.boundingRect(lime_green_mask)
        x, y, w, h = green_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 205, 50), 2)
        cv2.putText(frame, "Lime Green", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 205, 50), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("yellow_mask", yellow_mask)
    cv2.imshow("lime_green_mask", lime_green_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser is not None:
    ser.close()