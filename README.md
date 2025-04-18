

# Hand Gesture Volume Control using MediaPipe and OpenCV

This project uses OpenCV and MediaPipe to control the system volume on macOS based on hand gestures. By detecting the distance between the thumb and index finger, the system adjusts the volume in real-time.

## 💾 Requirements
To get started, install the necessary dependencies using `pip`:

```bash
pip install opencv-python mediapipe numpy
```

### macOS Volume Control
This project uses AppleScript to control the system volume. Ensure that you're running macOS for this to work.

## 📝 Code Explanation

### Importing Libraries

```python
import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess
```

### Volume Control Function (macOS Only)

This function uses AppleScript to set the system volume based on the distance between the thumb and index finger.

```python
# Volume Control Function for macOS
def set_volume(volume_level):
    # volume_level should be between 0 and 100
    applescript = f"osascript -e 'set volume output volume {volume_level}'"
    subprocess.run(applescript, shell=True)
```

### MediaPipe Hand Landmark Model

We use MediaPipe's Hand module to detect hand landmarks and calculate the distance between the thumb and index finger. This distance is used to control the volume.

```python
# solution APIs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
```

### Webcam Setup

Set up the webcam with OpenCV to capture live video.

```python
# Webcam Setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)
```

### Hand Detection and Volume Adjustment

Using MediaPipe's Hand module, the code detects hand landmarks and calculates the distance between the thumb and index finger. The `set_volume()` function adjusts the system volume based on this distance.

```python
# Mediapipe Hand Landmark Model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        success, image = cam.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Finding position of Hand landmarks      
        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])          

        # Assigning variables for Thumb and Index finger position
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Marking Thumb and Index finger
            cv2.circle(image, (x1, y1), 15, (255, 255, 255))  
            cv2.circle(image, (x2, y2), 15, (255, 255, 255))   
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            vol = np.interp(length, [50, 220], [0, 100])
            set_volume(vol)  # Set volume using the AppleScript function

            volBar = np.interp(length, [50, 220], [400, 150])
            volPer = np.interp(length, [50, 220], [0, 100])

            # Volume Bar
            cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 0), 3)
    
        cv2.imshow('handDetector', image) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cam.release()
```

### How it Works:

- The webcam captures the video and processes it frame by frame using MediaPipe.
- The hand landmarks are detected and used to calculate the distance between the thumb and index finger.
- This distance is then mapped to a volume range (0-100) and the volume is adjusted accordingly on macOS using AppleScript.
- A volume bar is displayed on the screen, reflecting the current volume percentage.

## 💻 Compatibility

This version of the project is specifically designed for macOS, as it uses AppleScript for volume control. It will not work on other operating systems like Windows or Linux without modification.

---
