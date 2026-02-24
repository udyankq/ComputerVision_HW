import cv2
import numpy as np

# Загружаем изображение
img = cv2.imread('HW4_Harris/assets/my_object.jpg')

# Переводим в серый
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris detector
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Немного усиливаем углы
dst = cv2.dilate(dst, None)

# Отмечаем углы красным
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow("Harris Corners", img)
cv2.waitKey(0)
cv2.destroyAllWindows()