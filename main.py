import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess
import pycaw
from collections import deque
import time

# Swipe gesture detection history
wrist_x_history = deque(maxlen=10)
swipe_cooldown = 1  # seconds
last_swipe_time = 0

def next_track():
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to next track'])

def previous_track():
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to previous track"'])

# Volume Control Function for macOS
def set_volume(volume_level):
    applescript = f"osascript -e 'set volume output volume {int(volume_level)}'"
    subprocess.run(applescript, shell=True)

# Store volume state
last_known_volume = 50
current_volume = -1  # to prevent spamming set_volume()

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam Setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lmList = []

        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # ---- SWIPE GESTURE DETECTION ----
            wrist_x = lmList[0][1]
            wrist_x_history.append(wrist_x)

            if len(wrist_x_history) >= 2:
                delta_x = wrist_x_history[-1] - wrist_x_history[0]
                current_time = time.time()

                if abs(delta_x) > 80 and (current_time - last_swipe_time) > swipe_cooldown:
                    if delta_x > 0:
                        next_track()
                        print("Swipe Right → Next Track")
                    else:
                        previous_track()
                        print("Swipe Left → Previous Track")
                    last_swipe_time = current_time

            # ---- VOLUME CONTROL ----
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            length = math.hypot(x2 - x1, y2 - y1)

            cv2.circle(image, (x1, y1), 15, (255, 255, 255))
            cv2.circle(image, (x2, y2), 15, (255, 255, 255))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            vol = np.interp(length, [50, 220], [0, 100])
            volBar = np.interp(length, [50, 220], [400, 150])
            volPer = int(np.interp(length, [50, 220], [0, 100]))

            if length < 50:
                if current_volume != 0:
                    set_volume(0)
                    current_volume = 0
                cv2.putText(image, "Muted", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            else:
                if abs(volPer - current_volume) > 5:
                    set_volume(volPer)
                    current_volume = volPer
                    last_known_volume = volPer

                cv2.putText(image, f'Volume: {volPer}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

            # Volume Bar UI
            cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)

        else:
            cv2.putText(image, "No Hand Detected", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 100, 100), 2)

        cv2.imshow('handDetector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cam.release()
