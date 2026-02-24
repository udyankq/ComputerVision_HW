import cv2

img1 = cv2.imread("HW11_Stitching/assets/my_object4.jpg")
img2 = cv2.imread("HW11_Stitching/assets/my_object5.jpg")

if img1 is None or img2 is None:
    raise FileNotFoundError("Images not found")

stitcher = cv2.Stitcher_create()

status, result = stitcher.stitch([img1, img2])

if status == cv2.Stitcher_OK:
    cv2.imshow("Panorama", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Stitching failed. Status:", status)