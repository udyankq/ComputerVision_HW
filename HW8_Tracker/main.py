import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Camera frame not captured")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
eq = cv2.equalizeHist(gray)

hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(eq, cmap="gray")
plt.title("Equalized")
plt.axis("off")

plt.subplot(2, 1, 2)
plt.title("Histograms (bins=256, mask=off)")
plt.plot(hist_gray)
plt.plot(hist_eq)
plt.xlim([0, 256])

plt.tight_layout()
plt.show()