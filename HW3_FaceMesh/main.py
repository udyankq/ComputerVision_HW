import cv2
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

FACE_BOUNDS = [10, 152, 234, 454]
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]

def draw_bbox(frame, face, ids, color):
    xs = [face[i][0] for i in ids]
    ys = [face[i][1] for i in ids]
    cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]
        draw_bbox(frame, face, FACE_BOUNDS, (0, 255, 0))
        draw_bbox(frame, face, LEFT_EYE, (0, 0, 255))
        draw_bbox(frame, face, RIGHT_EYE, (255, 0, 0))

    cv2.imshow("HW3 FaceMesh (q)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()