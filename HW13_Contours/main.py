import cv2
import numpy as np

img_path = "HW13_Contours/assets/my_object6.jpg"
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 50, 150)

kernel = np.ones((3, 3), np.uint8)
edged_thick = cv2.dilate(edged, kernel, iterations=1)

cnts, _ = cv2.findContours(edged_thick.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)

cv2.imshow("coins1 - contours", coins)
cv2.imshow("coins2 - edges", edged_thick)

cv2.waitKey(0)
cv2.destroyAllWindows()