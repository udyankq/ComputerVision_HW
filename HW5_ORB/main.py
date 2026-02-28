import cv2

img = cv2.imread('HW5_ORB/assets/my_object2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints = orb.detect(gray, None)

img_with_kp = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0))

cv2.imshow("ORB Features", img_with_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
