import cv2
import numpy as np

img_path = "HW10_Threshold/assets/my_object3.jpg"
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError("Image not found")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", image)
cv2.imshow("Thresholded", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
