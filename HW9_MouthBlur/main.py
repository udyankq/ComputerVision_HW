import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
blur_strength = 57

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        y1 = y + int(h * 0.6)
        y2 = y + h
        x1 = x + int(w * 0.2)
        x2 = x + int(w * 0.8)

        roi = frame[y1:y2, x1:x2]
        if roi.size != 0:
            roi_blur = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
            frame[y1:y2, x1:x2] = roi_blur

    cv2.imshow("Mouth Blur", frame)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()