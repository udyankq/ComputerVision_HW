import cv2
import numpy as np
import PoseModule as pm

VIDEO_PATH = "HW14_PushupCounter/assets/my_video2.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Can't open video: {VIDEO_PATH}")

detector = pm.poseDetector()

count = 0
direction = 0
form = 0
feedback = "Fix Form"

ELBOW_LOW = 90
ELBOW_HIGH = 165

while True:
    ret, img = cap.read()
    if not ret:
        cv2.waitKey(0)
        break

    img = detector.findPose(img, draw=True)

    angle_left = detector.findAngle(img, 11, 13, 15, draw=False)
    angle_right = detector.findAngle(img, 12, 14, 16, draw=False)

    elbow = None
    if angle_left is not None and angle_right is not None:
        elbow = angle_left if abs(angle_left - 140) < abs(angle_right - 140) else angle_right
    else:
        elbow = angle_left if angle_left is not None else angle_right

    if elbow is not None:
        per = np.interp(elbow, (ELBOW_LOW, ELBOW_HIGH), (0, 100))
        bar = np.interp(elbow, (ELBOW_LOW, ELBOW_HIGH), (380, 50))

        if elbow > (ELBOW_HIGH - 5):
            form = 1

        if form == 1:
            if per <= 10:
                feedback = "Up"
                if direction == 0:
                    count += 0.5
                    direction = 1

            if per >= 90:
                feedback = "Down"
                if direction == 1:
                    count += 0.5
                    direction = 0

        cv2.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
        cv2.rectangle(img, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f"{int(per)}%", (560, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.rectangle(img, (0, 380), (120, 480), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (30, 455), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        cv2.rectangle(img, (450, 0), (640, 50), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, feedback, (460, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.putText(img, f"Elbow: {int(elbow)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Pushup Counter (video)", img)

    key = cv2.waitKey(20) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()